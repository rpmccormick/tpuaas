"""Tier 1 Local Voice Pipeline + Tier 2 Capped Gemini Flash-Lite Fallback.

Runs 100% on the Intel Core i3-N300 CPU/iGPU (0.0% Coral EdgeTPU load):
1. Receives 16 kHz mono S16_LE PCM stream on UDP :8092 (16-byte header `MCBV<room padded to 12B>` + 1600B PCM).
2. Runs streaming VAD + configurable wake-word detection + faster-whisper (tiny.en int8) STT.
3. Routes transcripts through a Tier 1 Local Intent Matcher ($0, <10 ms) for all party/house commands
   (`party_mode_start`, `party_mode_stop`, `ambiled_pct`, `full_house_hue`, `summon_loona`, `visual_mode`).
4. Publishes structured intents to MQTT `<TPUAAS_TOPIC_PREFIX>/voice/intent` and synthesizes audio replies
   (`<TPUAAS_TOPIC_PREFIX>/voice/tts/<room>` + `GET /v1/voice/tts/<room>.wav`).
5. Routes unmatched open-ended queries to `gemini-2.5-flash-lite` with `maxOutputTokens=150` and a
   hard in-memory + disk-persisted counter enforcing `GEMINI_DAILY_CAP=50` calls/day.
"""
from __future__ import annotations

import base64
import collections
import io
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_WAKE_WORDS = ("atlas", "hey atlas", "okay atlas")


def _mqtt_enc_str(b: bytes) -> bytes:
    return struct.pack("!H", len(b)) + b


def _mqtt_enc_varlen(length: int) -> bytes:
    out = bytearray()
    while True:
        byte = length % 128
        length //= 128
        if length > 0:
            byte |= 0x80
        out.append(byte)
        if length == 0:
            break
    return bytes(out)


def mqtt_publish_one(
    host: str,
    port: int,
    user: str,
    password: str,
    topic: str,
    payload: bytes,
    timeout: float = 3.0,
) -> bool:
    """Zero-dependency MQTT 3.1.1 QoS-0 publisher."""
    if not host:
        return False
    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        cid = f"tpuaas-vpub-{uuid.uuid4().hex[:6]}".encode("utf-8")
        if user:
            var_hdr = b"\x00\x04MQTT\x04\xc2\x00\x1e"
            conn_pay = _mqtt_enc_str(cid) + _mqtt_enc_str(user.encode("utf-8")) + _mqtt_enc_str(password.encode("utf-8"))
        else:
            var_hdr = b"\x00\x04MQTT\x04\x02\x00\x1e"
            conn_pay = _mqtt_enc_str(cid)
        rem = var_hdr + conn_pay
        sock.sendall(b"\x10" + _mqtt_enc_varlen(len(rem)) + rem)
        connack = sock.recv(4)
        if len(connack) < 4 or connack[3] != 0:
            return False

        pub_rem = _mqtt_enc_str(topic.encode("utf-8")) + payload
        sock.sendall(b"\x30" + _mqtt_enc_varlen(len(pub_rem)) + pub_rem)
        sock.sendall(b"\xe0\x00")  # DISCONNECT
        return True
    except Exception:
        return False
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass


@dataclass
class RoomAudioState:
    room: str
    preroll: Deque[bytes] = field(default_factory=lambda: collections.deque(maxlen=16))
    speech_buf: bytearray = field(default_factory=bytearray)
    speaking: bool = False
    above_count: int = 0
    silence_count: int = 0
    noise_floor: float = 120.0
    packets_received: int = 0
    utterances_segmented: int = 0
    last_rms: float = 0.0
    last_packet_ts: float = 0.0
    followup_until: float = 0.0
    active_speaker_ip: str = ""
    active_speaker_ts: float = 0.0
    last_transcript: str = ""
    last_stt_ms: float = 0.0


class LocalVoiceEngine:
    def __init__(
        self,
        base_dir: Path,
        data_dir: Path,
        on_party_state_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.on_party_state_change = on_party_state_change

        self.udp_port = int(os.environ.get("TPUAAS_VOICE_UDP_PORT", "8092"))
        self.public_base_url = os.environ.get("TPUAAS_URL", "http://localhost:8090").rstrip("/")
        self.assistant_name = os.environ.get("TPUAAS_ASSISTANT_NAME", "Atlas").strip() or "Atlas"
        self.topic_prefix = os.environ.get("TPUAAS_TOPIC_PREFIX", "tpuaas").strip("/") or "tpuaas"

        raw_wakes = os.environ.get("TPUAAS_WAKE_WORDS", ",".join(DEFAULT_WAKE_WORDS))
        parsed_wakes = tuple(w.strip().lower() for w in raw_wakes.split(",") if w.strip())
        self.wake_word_aliases: Tuple[str, ...] = parsed_wakes or DEFAULT_WAKE_WORDS

        extra_strip = os.environ.get("TPUAAS_WAKE_STRIP_TOKENS", "")
        strip_set = {"hey", "ok", "okay", "yo", "hi"}
        for alias in self.wake_word_aliases:
            strip_set.update(re.findall(r"[a-z0-9']+", alias.lower()))
        for tok in extra_strip.split(","):
            if tok.strip():
                strip_set.update(re.findall(r"[a-z0-9']+", tok.strip().lower()))
        self.wake_strip_tokens = frozenset(strip_set)

        # MQTT Configuration (from environment / .env)
        self.mqtt_host = os.environ.get("TPUAAS_MQTT_HOST", "").strip()
        self.mqtt_port = int(os.environ.get("TPUAAS_MQTT_PORT", "1883"))
        self.mqtt_user = os.environ.get("TPUAAS_MQTT_USER", "").strip()
        self.mqtt_pass = os.environ.get("TPUAAS_MQTT_PASS", "")

        # Tier 2 Capped Gemini Flash-Lite Configuration
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite"
        self.gemini_max_output_tokens = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "150"))
        self.gemini_daily_cap = int(os.environ.get("GEMINI_DAILY_CAP", "50"))

        self._gemini_lock = threading.Lock()
        self.gemini_budget_file = self.data_dir / "gemini_budget.json"
        self.gemini_day = time.strftime("%Y-%m-%d")
        self.gemini_calls_today = 0
        self._load_gemini_budget()

        # Piper TTS paths
        self.piper_bin = self.base_dir / "piper" / "piper"
        self.piper_model = self.base_dir / "piper" / "en_US-lessac-medium.onnx"

        # Whisper STT & openWakeWord state (0% Coral TPU)
        self._whisper_lock = threading.Lock()
        self.whisper_model: Any = None
        self.whisper_loaded = False
        self.openwakeword_available = False

        # Per-room UDP audio stream states & history
        self.rooms: Dict[str, RoomAudioState] = {}
        self.recent_utterances: Deque[Dict[str, Any]] = collections.deque(maxlen=25)
        self.recent_transcripts: Deque[Dict[str, Any]] = collections.deque(maxlen=20)
        self.loona_summon_state: Dict[str, Any] = {
            "active": False,
            "action": "idle",
            "target_room": "living_room",
            "dance": False,
            "scene_class": "auto",
            "updated_at": 0.0,
        }

        self._stop_event = threading.Event()
        self._audio_queue: "collections.deque[Tuple[str, bytes]]" = collections.deque(maxlen=8)
        self._audio_cond = threading.Condition()

        # Start background model loader, UDP :8092 listener, and STT worker threads
        threading.Thread(target=self._init_voice_models, name="Voice-Model-Init", daemon=True).start()
        self.udp_thread = threading.Thread(target=self._udp_listener_loop, name="Voice-UDP-8092", daemon=True)
        self.udp_thread.start()
        self.stt_thread = threading.Thread(target=self._stt_worker_loop, name="Voice-STT-Worker", daemon=True)
        self.stt_thread.start()

    def _load_gemini_budget(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if self.gemini_budget_file.is_file():
            try:
                data = json.loads(self.gemini_budget_file.read_text(encoding="utf-8"))
                if data.get("date") == today:
                    self.gemini_day = today
                    self.gemini_calls_today = int(data.get("calls", 0))
                    return
            except Exception:
                pass
        self.gemini_day = today
        self.gemini_calls_today = 0
        self._save_gemini_budget()

    def _save_gemini_budget(self) -> None:
        try:
            self.gemini_budget_file.write_text(
                json.dumps(
                    {
                        "date": self.gemini_day,
                        "calls": self.gemini_calls_today,
                        "daily_cap": self.gemini_daily_cap,
                        "model": self.gemini_model,
                        "max_output_tokens": self.gemini_max_output_tokens,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _init_voice_models(self) -> None:
        try:
            import importlib.util
            self.openwakeword_available = importlib.util.find_spec("openwakeword") is not None
        except Exception:
            self.openwakeword_available = False

        try:
            from faster_whisper import WhisperModel
            whisper_dir = self.base_dir / "models" / "whisper"
            whisper_dir.mkdir(parents=True, exist_ok=True)
            model = WhisperModel(
                "tiny.en",
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                download_root=str(whisper_dir),
            )
            with self._whisper_lock:
                self.whisper_model = model
                self.whisper_loaded = True
        except Exception:
            self.whisper_loaded = False

    def get_status(self) -> Dict[str, Any]:
        with self._gemini_lock:
            today = time.strftime("%Y-%m-%d")
            if self.gemini_day != today:
                self.gemini_day = today
                self.gemini_calls_today = 0
                self._save_gemini_budget()
            calls_today = self.gemini_calls_today

        return {
            "udp_port": self.udp_port,
            "processor": "Intel Core i3-N300 CPU/iGPU (0.0% Coral EdgeTPU load)",
            "whisper_stt_loaded": self.whisper_loaded,
            "whisper_model": "tiny.en (int8)",
            "openwakeword_installed": self.openwakeword_available,
            "voice_tts_mode": "quiet_beep_only (spoken TTS disabled)",
            "piper_tts_ready": False,
            "gemini_tier2": {
                "configured": bool(self.gemini_api_key),
                "model": self.gemini_model,
                "max_output_tokens": self.gemini_max_output_tokens,
                "daily_cap": self.gemini_daily_cap,
                "calls_today": calls_today,
                "remaining_today": max(0, self.gemini_daily_cap - calls_today),
                "date": self.gemini_day,
            },
            "loona_party_summon": self.loona_summon_state,
            "rooms": {
                rm: {
                    "packets_received": st.packets_received,
                    "utterances_segmented": st.utterances_segmented,
                    "speaking": st.speaking,
                    "last_rms": round(st.last_rms, 1),
                    "noise_floor": round(st.noise_floor, 1),
                    "last_packet_ts": round(st.last_packet_ts, 2),
                    "last_transcript": st.last_transcript,
                    "last_stt_ms": st.last_stt_ms,
                }
                for rm, st in self.rooms.items()
            },
            "recent_utterances": list(self.recent_utterances)[:10],
            "recent_transcripts": list(self.recent_transcripts)[:15],
        }

    # --- UDP :8092 16 kHz Mono S16_LE Receiver (`MCBV<room>\x00` + 1600B PCM) ---

    def _udp_listener_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self.udp_port))
        except Exception:
            return
        sock.settimeout(1.0)

        while not self._stop_event.is_set():
            try:
                pkt, addr = sock.recvfrom(8192)
            except socket.timeout:
                continue
            except Exception:
                break

            if not pkt:
                continue

            if len(pkt) > 16 and pkt[:4] == b"MCBV":
                raw_room = pkt[4:16].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
                room = raw_room or "living_room"
                pcm = pkt[16:]
            else:
                room = "living_room"
                pcm = pkt

            if len(pcm) < 64:
                continue

            sender_ip = addr[0] if isinstance(addr, tuple) and addr else ""
            self._ingest_pcm_chunk(room, pcm, sender_ip=sender_ip)

    def _ingest_pcm_chunk(self, room: str, pcm: bytes, sender_ip: str = "") -> None:
        st = self.rooms.get(room)
        if st is None:
            st = RoomAudioState(room=room)
            self.rooms[room] = st

        now = time.time()
        st.packets_received += 1
        st.last_packet_ts = now

        # Ensure even byte count for S16_LE
        if len(pcm) % 2 != 0:
            pcm = pcm[:-1]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size > 0 else 0.0
        st.last_rms = rms

        thresh = max(550.0, min(1800.0, st.noise_floor * 3.5))
        if rms >= thresh:
            st.active_speaker_ip = sender_ip
            st.active_speaker_ts = now
            st.above_count += 1
            st.silence_count = 0
            if not st.speaking:
                if st.above_count >= 4:
                    st.speaking = True
                    st.speech_buf.clear()
                    for pre in st.preroll:
                        st.speech_buf.extend(pre)
                    st.preroll.clear()
                    st.speech_buf.extend(pcm)
                else:
                    st.preroll.append(pcm)
            else:
                st.speech_buf.extend(pcm)
                # Cap max single utterance at 8.0 seconds (256,000 bytes @ 16kHz S16_LE)
                if len(st.speech_buf) >= 256000:
                    self._flush_room_utterance(st)
        else:
            # If another sender IP is actively speaking in this room, ignore idle packets from secondary IPs
            if st.active_speaker_ip and sender_ip and sender_ip != st.active_speaker_ip and (now - st.active_speaker_ts) < 2.0:
                return
            st.above_count = 0
            if st.speaking:
                st.speech_buf.extend(pcm)
                st.silence_count += 1
                # ~0.70s of trailing silence (14 * 50ms chunks) completes the utterance
                if st.silence_count >= 14:
                    self._flush_room_utterance(st)
            else:
                st.noise_floor = 0.96 * st.noise_floor + 0.04 * max(40.0, rms)
                st.preroll.append(pcm)

    def _flush_room_utterance(self, st: RoomAudioState) -> None:
        raw_pcm = bytes(st.speech_buf)
        st.speech_buf.clear()
        st.speaking = False
        st.above_count = 0
        st.silence_count = 0
        # Require at least 0.60 seconds of audio (19,200 bytes @ 16kHz S16_LE) so standalone wake-word (~0.75s) works while clicks are ignored
        if len(raw_pcm) < 19200:
            return
        st.utterances_segmented += 1
        with self._audio_cond:
            self._audio_queue.append((st.room, raw_pcm))
            self._audio_cond.notify()

    def _stt_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._audio_cond:
                while not self._audio_queue and not self._stop_event.is_set():
                    self._audio_cond.wait(timeout=0.5)
                if self._stop_event.is_set():
                    return
                room, raw_pcm = self._audio_queue.popleft()

            try:
                st = self.rooms.get(room)
                in_followup = bool(st and time.time() < st.followup_until)
                self.process_pcm_utterance(
                    raw_pcm=raw_pcm,
                    room=room,
                    require_wake_word=not in_followup,
                    source="udp_8092",
                )
            except Exception:
                pass

    # --- Speech-to-Text (`faster-whisper` on CPU) & Wake-Word Detection ---

    def transcribe_pcm_s16le(self, raw_pcm: bytes, sample_rate: int = 16000) -> Tuple[str, float]:
        t0 = time.perf_counter()
        if len(raw_pcm) % 2 != 0:
            raw_pcm = raw_pcm[:-1]
        audio_f32 = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != 16000 and audio_f32.size > 0:
            # Linear resample to 16 kHz if caller uploaded non-16kHz WAV
            duration = audio_f32.size / float(sample_rate)
            target_len = max(1, int(round(duration * 16000)))
            x_old = np.linspace(0.0, 1.0, num=audio_f32.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            audio_f32 = np.interp(x_new, x_old, audio_f32).astype(np.float32)

        with self._whisper_lock:
            if self.whisper_model is None:
                self._init_voice_models()
            model = self.whisper_model

        if model is None:
            return "", 0.0

        # Use hotwords=self.assistant_name (not initial_prompt!) with Silero VAD so real speech saying the wake word
        # is recognized accurately without hallucinating on silence or clicks.
        segments, _info = model.transcribe(
            audio_f32,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            no_speech_threshold=0.45,
            hotwords=self.assistant_name,
        )
        valid_segs = [
            seg.text.strip()
            for seg in segments
            if getattr(seg, "no_speech_prob", 0.0) < 0.50 and getattr(seg, "avg_logprob", 0.0) > -0.95
        ]
        text = " ".join(valid_segs).strip()
        stt_ms = (time.perf_counter() - t0) * 1000.0
        return text, round(stt_ms, 1)

    def detect_and_strip_wake_word(self, transcript: str) -> Tuple[bool, Optional[str], str]:
        """Check for configured wake word (and phonetic STT variants) and return (detected, matched_alias, clean_command)."""
        norm = transcript.strip()
        lower = norm.lower()
        for alias in self.wake_word_aliases:
            idx = lower.find(alias)
            if idx != -1:
                before = norm[:idx].rstrip(" ,.:;!?-")
                # Strip optional greeting before wake word ("hey", "ok", "okay", "yo")
                before = re.sub(r"^(?:hey|ok|okay|yo|hi)\b[\s,.:;!?-]*", "", before, flags=re.IGNORECASE).strip()
                after = norm[idx + len(alias) :].lstrip(" ,.:;!?-")
                combined = f"{before} {after}".strip()
                clean = combined if combined else norm
                return True, alias, clean
        return False, None, norm

    # --- Tier 1 Local Intent Matcher ($0, <10 ms, 0 API Calls) ---

    def match_tier1_intent(self, clean_command: str, full_transcript: str) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()
        text = f"{clean_command} {full_transcript}".lower().strip()
        if not text:
            return None

        # 1. Check for Party / AmbiLED / Music Stop commands first
        stop_patterns = (
            r"\b(stop|stopping|end|ending|kill|shut\s*down|cancel|exit|quit)\b.*\b(party|music|video|videos|ambiled|ambi\s*led|and\s*by\s*led|light\s*show|show|dancing|dance)\b",
            r"\b(party\s*mode|ambiled|ambi\s*led|tv\s*ambiled|full\s*house\s*hue|light\s*show)\s*(off|stop|disable)\b",
            r"\b(turn\s*off|turning\s*off|switch\s*off|disable)\b.*\b(party|ambiled|ambi\s*led|and\s*by\s*led|tv|full\s*house\s*hue|light\s*show|music\s*videos?)\b",
            r"\b(normal\s*lights|restore\s*lights|restoring\s*warm|regular\s*lights|good\s*night|goodnight)\b",
        )
        for pat in stop_patterns:
            if re.search(pat, text):
                return {
                    "intent": "party_mode_stop",
                    "ambiled_pct": 0,
                    "full_house_hue": False,
                    "summon_loona": False,
                    "visual_mode": "none",
                    "reply_text": f"Stopping {self.assistant_name} party mode, turning off TV AmbiLED, and restoring warm room lights.",
                    "tier": "tier1_local",
                    "intent_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        # 2. Universal YouTube & Media Cast Matcher (Multi-purpose everyday use)
        #    e.g. "play Chris Stapleton Tiny Desk on the Living Room TV"
        #         "cast 90s rock to the office speaker"
        #         "watch Mark Rober on the office display"
        cast_match = re.search(
            r"\b(?:play|cast|put\s+on|watch|stream|queue)\s+(.+?)(?:\s+\b(?:on|to|in)\s+(?:the\s+)?(living\s+room(?:\s+tv)?|tv|main\s+tv|oled|google\s+tv|office\s+display|office\s+speaker|office\s+hub|office|master\s+display|master\s+bedroom|master\s+hub|bedroom\s+display|family\s+hub|kitchen\s+screen|echo\s+hub|kitchen|kodi|ashley'?s?\s+room|ashley))?(?:\s+with\s+(ambiled|ambi\s*led))?$",
            clean_command.strip(),
            flags=re.IGNORECASE,
        )
        if cast_match:
            raw_q = cast_match.group(1).strip(" ,.:;!?-")
            raw_target = (cast_match.group(2) or "living_room_tv").strip(" ,.:;!?-").lower()
            with_ambi = bool(cast_match.group(3)) or ("with ambiled" in text or "with ambi led" in text)
            raw_q = re.sub(r"\s+\b(?:with\s+ambi\s*led|on\s+youtube|from\s+youtube)\b.*$", "", raw_q, flags=re.IGNORECASE).strip()
            raw_q = re.sub(
                r"\s+\b(?:on|to|in)\s+(?:the\s+)?(living\s+room(?:\s+tv)?|tv|main\s+tv|oled|google\s+tv|office\s+display|office\s+speaker|office\s+hub|office|master\s+display|master\s+bedroom|master\s+hub|bedroom\s+display|family\s+hub|kitchen\s+screen|echo\s+hub|kitchen|kodi|ashley'?s?\s+room|ashley)$",
                "",
                raw_q,
                flags=re.IGNORECASE,
            ).strip()
            q_low = raw_q.lower()
            is_party_phrase = any(
                p in q_low
                for p in (
                    "favorite music",
                    "music video",
                    "party mode",
                    "dance party",
                    "the party",
                    "milkdrop",
                    "milk drop",
                    "projectm",
                    "loona",
                    "full house hue",
                )
            )
            if raw_q and not is_party_phrase:
                return {
                    "intent": "av_cast_youtube",
                    "cast_query": raw_q,
                    "cast_target": re.sub(r"[^a-z0-9]+", "_", raw_target).strip("_") or "living_room_tv",
                    "ambiled_pct": 75 if with_ambi else 0,
                    "full_house_hue": False,
                    "summon_loona": False,
                    "visual_mode": "youtube_cast",
                    "reply_text": "[quiet_good_beep]",
                    "tier": "tier1_local",
                    "intent_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        # 3. Check for Party Start / Music Videos / MilkDrop / AmbiLED / Full House Hue / Loona Dance commands
        party_triggers = (
            "music video",
            "music videos",
            "favorite music",
            "party mode",
            "dance party",
            "start the party",
            "let's party",
            "lets party",
            "dance with us",
            "drive loona",
            "summon loona",
            "bring loona",
            "milkdrop",
            "milk drop",
            "projectm",
            "project m",
            "ambiled",
            "ambi led",
            "rpmlight",
            "full house hue",
            "house hue",
        )
        if any(k in text for k in party_triggers):
            # Extract ambiled_pct (default 75% for music_video mode)
            ambiled_pct = 75
            pct_match = re.search(r"\b(\d{1,3})\s*(?:%|percent)\b", text)
            if pct_match:
                ambiled_pct = max(1, min(100, int(pct_match.group(1))))
            elif any(w in text for w in ("100%", "max ambiled", "maximum ambiled", "full brightness")):
                ambiled_pct = 100
            elif any(w in text for w in ("subtle", "gentle", "low ambiled", "dim ambiled")):
                ambiled_pct = 50

            # Extract full_house_hue (default True unless negated)
            if any(neg in text for neg in ("no hue", "without hue", "without full house", "tv only", "just the tv")):
                full_house_hue = False
            else:
                full_house_hue = True

            # Extract summon_loona (default True unless negated)
            if any(neg in text for neg in ("no loona", "without loona", "leave loona", "don't drive loona", "dont drive loona")):
                summon_loona = False
            else:
                summon_loona = True

            # Extract visual_mode ("music_videos" or "milkdrop")
            if any(m in text for m in ("milkdrop", "milk drop", "projectm", "project m", "visualizer")):
                visual_mode = "milkdrop"
            else:
                visual_mode = "music_videos"

            vis_label = "MilkDrop visualizer" if visual_mode == "milkdrop" else "your favorite music videos"
            parts = [f"playing {vis_label}", f"AmbiLED at {ambiled_pct} percent"]
            if full_house_hue:
                parts.append("Full House Hue stage lights")
            if summon_loona:
                parts.append("driving Loona in to dance")
            reply_text = f"{self.assistant_name} Party Mode engaged: " + ", ".join(parts) + "."

            return {
                "intent": "party_mode_start",
                "ambiled_pct": ambiled_pct,
                "full_house_hue": full_house_hue,
                "summon_loona": summon_loona,
                "visual_mode": visual_mode,
                "reply_text": reply_text,
                "tier": "tier1_local",
                "intent_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        # 4. Check for local system status query
        if any(k in text for k in ("system status", "tpu status", "tpu service", "how are you doing", "status report")):
            rem = max(0, self.gemini_daily_cap - self.gemini_calls_today)
            return {
                "intent": "system_status",
                "ambiled_pct": 75,
                "full_house_hue": False,
                "summon_loona": False,
                "visual_mode": "music_videos",
                "reply_text": f"{self.assistant_name} TPU service is online with {rem} of {self.gemini_daily_cap} daily cloud queries remaining.",
                "tier": "tier1_local",
                "intent_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        return None

    # --- Tier 2 Capped Gemini Flash-Lite Fallback (Hard 50 Calls/Day Cap, 150 Max Output Tokens) ---

    def query_tier2_gemini(self, clean_command: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with self._gemini_lock:
            today = time.strftime("%Y-%m-%d")
            if self.gemini_day != today:
                self.gemini_day = today
                self.gemini_calls_today = 0

            if not self.gemini_api_key:
                return {
                    "intent": "general_chat",
                    "reply_text": f"Local {self.assistant_name} voice is online, but no Gemini API key is configured for open-ended questions.",
                    "tier": "tier2_unconfigured",
                    "gemini_calls_today": self.gemini_calls_today,
                    "gemini_daily_cap": self.gemini_daily_cap,
                    "intent_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            if self.gemini_calls_today >= self.gemini_daily_cap:
                return {
                    "intent": "general_chat",
                    "reply_text": f"My daily cloud limit of {self.gemini_daily_cap} calls has been reached. All local {self.assistant_name} party and house commands remain available.",
                    "tier": "tier2_capped_blocked",
                    "gemini_calls_today": self.gemini_calls_today,
                    "gemini_daily_cap": self.gemini_daily_cap,
                    "intent_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            # Increment hard counter BEFORE network call so concurrent requests can never exceed GEMINI_DAILY_CAP
            self.gemini_calls_today += 1
            call_num = self.gemini_calls_today
            self._save_gemini_budget()

        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            f"You are {self.assistant_name}, the witty, concise smart-home show director for the Living Room. "
                            "Answer in 1 or 2 short spoken sentences suitable for text-to-speech (no markdown or bullet points)."
                        )
                    }
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": clean_command}]}],
            "generationConfig": {
                "maxOutputTokens": self.gemini_max_output_tokens,
                "temperature": 0.4,
            },
        }

        reply_text = ""
        tier_status = "tier2_gemini"
        candidate_models = [self.gemini_model]
        for fallback_m in ("gemini-3.5-flash-lite", "gemini-flash-lite-latest"):
            if fallback_m not in candidate_models:
                candidate_models.append(fallback_m)

        for model_id in candidate_models:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_id}:generateContent?key={self.gemini_api_key}"
            )
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=6.0) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    candidates = body.get("candidates") or []
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts") or []
                        reply_text = " ".join(p.get("text", "").strip() for p in parts if p.get("text")).strip()
                        if reply_text:
                            tier_status = "tier2_gemini"
                            break
            except Exception:
                tier_status = "tier2_gemini_fallback"
                continue

        if not reply_text:
            reply_text = (
                f"I heard you say: {clean_command}. Say {self.assistant_name} start party mode whenever you are ready for music videos and AmbiLED."
            )

        return {
            "intent": "general_chat",
            "ambiled_pct": 75,
            "full_house_hue": False,
            "summon_loona": False,
            "visual_mode": "music_videos",
            "reply_text": reply_text,
            "tier": tier_status,
            "gemini_model": self.gemini_model,
            "gemini_max_output_tokens": self.gemini_max_output_tokens,
            "gemini_calls_today": call_num,
            "gemini_daily_cap": self.gemini_daily_cap,
            "intent_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # --- Quiet Good-Beep / Bad-Beep Audio Feedback (All Spoken Voice TTS Disabled) ---

    def synthesize_tts_wav(self, text: str, room: str = "living_room", is_good_beep: bool = True) -> Tuple[Path, float]:
        """Generate a very quiet, gentle Good-Beep or Bad-Beep WAV (no spoken voice TTS)."""
        t0 = time.perf_counter()
        safe_room = re.sub(r"[^a-zA-Z0-9_-]", "_", room) or "living_room"
        out_path = self.data_dir / f"tts_{safe_room}.wav"
        latest_path = self.data_dir / "tts_latest.wav"

        sr = 16000
        # Gentle, crisp amplitude (~9.8% of full scale) in the 880-1175 Hz band where the tiny Brilliant speaker is audible
        amp = 3200.0
        if is_good_beep:
            # Gentle rising two-note blip (A5 -> D6, 75ms each)
            freqs = (880.0, 1174.66)
            dur = 0.075
        else:
            # Gentle low two-note blip (D5 -> A4, 80ms each)
            freqs = (587.33, 440.00)
            dur = 0.080

        pieces = []
        gap = np.zeros(int(sr * 0.025), dtype=np.int16)
        for f_hz in freqs:
            n = int(sr * dur)
            t_arr = np.linspace(0.0, dur, n, endpoint=False)
            env = np.hanning(n)
            tone = (np.sin(2.0 * np.pi * f_hz * t_arr) * env * amp).astype(np.int16)
            pieces.append(tone)
            pieces.append(gap)

        full_wave = np.concatenate(pieces)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(full_wave.tobytes())

        try:
            latest_path.write_bytes(out_path.read_bytes())
        except Exception:
            pass

        tts_ms = (time.perf_counter() - t0) * 1000.0
        return out_path, round(tts_ms, 1)

    # --- End-to-End Utterance Evaluation & MQTT Dispatch ---

    def handle_transcript(
        self,
        transcript: str,
        room: str = "living_room",
        require_wake_word: bool = False,
        stt_ms: float = 0.0,
        source: str = "http",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        wake_detected, matched_alias, clean_cmd = self.detect_and_strip_wake_word(transcript)

        if require_wake_word and not wake_detected:
            return {
                "accepted": False,
                "reason": "wake_word_not_detected",
                "room": room,
                "source": source,
                "transcript": transcript,
                "wake_word_detected": False,
                "stt_ms": stt_ms,
            }

        # Strip wake words / greetings to see if the user spoke a command in the same breath or just said the wake word
        cmd_words = [
            w for w in re.findall(r"[a-zA-Z0-9']+", clean_cmd.lower())
            if w not in self.wake_strip_tokens
        ]

        # Guard against repeated wake-word hallucination
        if wake_detected and matched_alias and transcript.lower().count(matched_alias) >= 2 and len(cmd_words) < 2:
            return {
                "accepted": False,
                "reason": "empty_command_after_wake_word",
                "room": room,
                "source": source,
                "transcript": transcript,
                "clean_command": clean_cmd,
                "wake_word_detected": wake_detected,
                "stt_ms": stt_ms,
            }

        # If the user just said the wake word to wake it up, play the Good Beep
        # and open a 6-second follow-up window for their command!
        if wake_detected and len(cmd_words) == 0:
            st = self.rooms.get(room)
            if st is not None and not dry_run:
                st.followup_until = time.time() + 6.0
            intent_res = {
                "intent": "wake_word_ping",
                "ambiled_pct": 0,
                "full_house_hue": False,
                "summon_loona": False,
                "visual_mode": "none",
                "reply_text": "[quiet_good_beep]",
                "tier": "tier1_local",
                "intent_ms": 0.05,
            }
            is_good_beep = True
        else:
            # 1. Try Tier 1 Local Intent Matcher ($0, <10 ms)
            intent_res = self.match_tier1_intent(clean_cmd, transcript)
            is_good_beep = True
            if intent_res is None:
                # Unrecognized command -> Spoken Gemini voice responses are disabled; emit a very quiet bad beep only.
                is_good_beep = False
                intent_res = {
                    "intent": "unrecognized_command",
                    "ambiled_pct": 75,
                    "full_house_hue": False,
                    "summon_loona": False,
                    "visual_mode": "music_videos",
                    "reply_text": "[quiet_bad_beep]",
                    "tier": "tier1_bad_beep",
                    "intent_ms": 0.01,
                }

        reply_text = "[quiet_good_beep]" if is_good_beep else "[quiet_bad_beep]"
        wav_path, tts_ms = self.synthesize_tts_wav(reply_text, room=room, is_good_beep=is_good_beep)
        ts_now = round(time.time(), 2)
        safe_room = re.sub(r"[^a-zA-Z0-9_-]", "_", room) or "living_room"
        tts_url = f"{self.public_base_url}/v1/voice/tts/{safe_room}.wav?ts={int(ts_now)}"

        # Update Loona party summon & TPUaaS lightshow state if party mode started/stopped (skip on dry_run)
        intent_name = intent_res["intent"]
        if not dry_run and intent_name == "party_mode_start":
            if intent_res.get("summon_loona", True):
                self.update_loona_summon(
                    {
                        "action": "party_summon",
                        "target_room": room,
                        "dance": True,
                        "scene_class": "music_video",
                    }
                )
            if self.on_party_state_change:
                self.on_party_state_change(
                    {
                        "lightshow_active": True,
                        "slider_pct": int(intent_res.get("ambiled_pct", 75)),
                        "scene_class": "music_video",
                        "beat_pulse": True,
                        "energy": 0.80,
                    }
                )
        elif not dry_run and intent_name == "party_mode_stop":
            self.update_loona_summon(
                {
                    "action": "party_stop",
                    "target_room": room,
                    "dance": False,
                    "scene_class": "auto",
                }
            )
            if self.on_party_state_change:
                self.on_party_state_change(
                    {
                        "lightshow_active": False,
                        "beat_pulse": False,
                        "energy": 0.0,
                        "scene_class": "auto",
                    }
                )

        av_cast_res = None
        if intent_name == "av_cast_youtube":
            try:
                cast_req = urllib.request.Request(
                    "http://127.0.0.1:8095/v1/av/cast",
                    data=json.dumps(
                        {
                            "query": intent_res.get("cast_query", ""),
                            "target": intent_res.get("cast_target", "living_room_tv"),
                            "ambiled": intent_res.get("ambiled_pct", 0),
                            "dry_run": dry_run,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(cast_req, timeout=5.0) as c_resp:
                    av_cast_res = json.loads(c_resp.read().decode("utf-8"))
            except Exception as e:
                av_cast_res = {"ok": False, "error": str(e)}

        # Build MQTT payload for `<topic_prefix>/voice/intent`
        intent_payload = {
            "intent": intent_name,
            "room": room,
            "source": source,
            "transcript": transcript,
            "clean_command": clean_cmd,
            "wake_word_detected": wake_detected,
            "wake_word_alias": matched_alias,
            "cast_query": intent_res.get("cast_query"),
            "cast_target": intent_res.get("cast_target"),
            "av_cast": av_cast_res,
            "ambiled_pct": intent_res.get("ambiled_pct", 75),
            "full_house_hue": intent_res.get("full_house_hue", True),
            "summon_loona": intent_res.get("summon_loona", True),
            "visual_mode": intent_res.get("visual_mode", "music_videos"),
            "reply_text": reply_text,
            "tts_url": tts_url,
            "tier": intent_res.get("tier", "tier1_local"),
            "stt_ms": stt_ms,
            "intent_ms": intent_res.get("intent_ms", 0.0),
            "tts_ms": tts_ms,
            "total_ms": round((time.perf_counter() - t_start) * 1000.0 + stt_ms, 1),
            "gemini_calls_today": self.gemini_calls_today,
            "gemini_daily_cap": self.gemini_daily_cap,
            "ts": ts_now,
        }

        tts_payload = {
            "room": room,
            "text": reply_text,
            "url": tts_url,
            "wav_url": tts_url,
            "intent": intent_name,
            "tier": intent_payload["tier"],
            "ts": ts_now,
        }

        intent_topic = f"{self.topic_prefix}/voice/intent/dry_run" if dry_run else f"{self.topic_prefix}/voice/intent"
        tts_topic = f"{self.topic_prefix}/voice/tts/dry_run_{safe_room}" if dry_run else f"{self.topic_prefix}/voice/tts/{safe_room}"
        mqtt_intent_ok = mqtt_publish_one(
            self.mqtt_host,
            self.mqtt_port,
            self.mqtt_user,
            self.mqtt_pass,
            intent_topic,
            json.dumps(intent_payload).encode("utf-8"),
        )
        mqtt_tts_ok = mqtt_publish_one(
            self.mqtt_host,
            self.mqtt_port,
            self.mqtt_user,
            self.mqtt_pass,
            tts_topic,
            json.dumps(tts_payload).encode("utf-8"),
        )

        intent_payload["accepted"] = True
        intent_payload["dry_run"] = dry_run
        intent_payload["mqtt_published"] = {
            "intent_topic": intent_topic,
            "intent_ok": mqtt_intent_ok,
            "tts_topic": tts_topic,
            "tts_ok": mqtt_tts_ok,
        }
        self.recent_utterances.appendleft(intent_payload)
        return intent_payload

    def process_pcm_utterance(
        self,
        raw_pcm: bytes,
        room: str = "living_room",
        sample_rate: int = 16000,
        require_wake_word: bool = False,
        source: str = "http_pcm",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        transcript, stt_ms = self.transcribe_pcm_s16le(raw_pcm, sample_rate=sample_rate)
        st = self.rooms.get(room)
        if st is not None:
            st.last_transcript = transcript
            st.last_stt_ms = stt_ms
        if transcript:
            self.recent_transcripts.appendleft(
                {
                    "room": room,
                    "transcript": transcript,
                    "stt_ms": stt_ms,
                    "pcm_bytes": len(raw_pcm),
                    "ts": round(time.time(), 2),
                }
            )
        if not transcript:
            return {
                "accepted": False,
                "reason": "empty_transcript",
                "room": room,
                "source": source,
                "stt_ms": stt_ms,
            }
        return self.handle_transcript(
            transcript=transcript,
            room=room,
            require_wake_word=require_wake_word,
            stt_ms=stt_ms,
            source=source,
            dry_run=dry_run,
        )

    def update_loona_summon(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action = str(payload.get("action", "party_summon"))
        self.loona_summon_state = {
            "active": action != "party_stop" and bool(payload.get("dance", True)),
            "action": action,
            "target_room": str(payload.get("target_room", "living_room")),
            "dance": bool(payload.get("dance", True)),
            "scene_class": str(payload.get("scene_class", "music_video")),
            "updated_at": round(time.time(), 2),
        }
        return dict(self.loona_summon_state)
