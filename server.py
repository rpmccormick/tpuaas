from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure self-contained EdgeTPU runtime is on sys.path before importing engine
RUNTIME_DIR = Path(os.environ.get("TPUAAS_RUNTIME_DIR", "/opt/tpuaas/runtime"))
if RUNTIME_DIR.exists() and str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

import ipaddress
import re
import socket
import urllib.parse
import wave

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

from engine import TPUEngine, decode_i420_to_pil
from voice import LocalVoiceEngine, mqtt_publish_one

_default_base = "/opt/tpuaas" if Path("/opt/tpuaas/models").exists() else str(Path(__file__).parent)
BASE_DIR = Path(os.environ.get("TPUAAS_BASE_DIR", _default_base))
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DASHBOARD_FILE = Path(__file__).parent / "dashboard.html"
GUIDE_FILE = Path(__file__).parent / "AGENT_GUIDE.md"
BOARD_FILE = Path(__file__).parent / "AGENT_BOARD.md"
ASSISTANT_NAME = os.environ.get("TPUAAS_ASSISTANT_NAME", "Atlas").strip() or "Atlas"
TOPIC_PREFIX = os.environ.get("TPUAAS_TOPIC_PREFIX", "tpuaas").strip("/") or "tpuaas"

SAMPLE_FILES = {
    "parrot": MODELS_DIR / "sample_parrot.jpg",
    "face": MODELS_DIR / "sample_face.bmp",
    "pose": MODELS_DIR / "sample_pose.bmp",
}

app = FastAPI(
    title=f"TPUaaS — Coral EdgeTPU-as-a-Service + {ASSISTANT_NAME} Voice Engine",
    description="Shared Multi-Project Hardware EdgeTPU Inference + i3-N300 Voice/STT/TTS Server for AmbiLED, Loona, Drones, and Camera Detections",
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = TPUEngine(models_dir=MODELS_DIR, data_dir=DATA_DIR)
voice_engine = LocalVoiceEngine(
    base_dir=BASE_DIR,
    data_dir=DATA_DIR,
    on_party_state_change=engine.update_lightshow_state,
)
engine.on_loona_command = voice_engine.update_loona_summon  # type: ignore


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    msg = str(exc)
    if "TPU_LOCKED_BY_EXCLUSIVE_LEASE" in msg:
        return JSONResponse(
            status_code=423,
            content={
                "error": "TPU_LOCKED_BY_EXCLUSIVE_LEASE",
                "detail": msg,
                "active_lease": engine.get_active_lease(),
                "cached_ambiled_cue": engine.stream_mailboxes.get("ambiled_cue"),
                "cached_lightshow_cue": engine.stream_mailboxes.get("lightshow_cue"),
            },
        )
    return JSONResponse(status_code=500, content={"error": "RuntimeError", "detail": msg})


def _decode_bytes_to_pil(
    raw: bytes,
    fmt: Optional[str] = None,
    width: int = 320,
    height: int = 176,
) -> Image.Image:
    expected_i420 = (width * height * 3) // 2
    if (fmt and fmt.lower() == "i420") or len(raw) == expected_i420 or len(raw) == 84480:
        w = width if len(raw) == expected_i420 else 320
        h = height if len(raw) == expected_i420 else 176
        return decode_i420_to_pil(raw, width=w, height=h)
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image bytes (len={len(raw)}, format={fmt}): {exc}",
        ) from exc


_BLOCKED_IMAGE_PORTS = frozenset({22, 23, 111, 1883, 3128, 3306, 3389, 3493, 5432, 6379, 8006})


def _validate_remote_image_url(raw_url: str) -> str:
    """Block SSRF against loopback, link-local/metadata, non-HTTP(S) schemes, and admin ports."""
    parsed = urllib.parse.urlparse(raw_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=400, detail="image_url must use http:// or https:// with a valid host")
    if parsed.port and parsed.port in _BLOCKED_IMAGE_PORTS:
        raise HTTPException(status_code=400, detail=f"image_url port {parsed.port} is restricted")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to resolve image_url host '{parsed.hostname}'") from exc
    for info in infos:
        addr_str = info[4][0].split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(addr_str)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast:
            raise HTTPException(status_code=400, detail="image_url resolves to a restricted local/metadata address")
    return raw_url.strip()


async def resolve_input_image(
    request: Request,
    file: Optional[UploadFile] = None,
    sample: Optional[str] = None,
    image_url: Optional[str] = None,
) -> Image.Image:
    fmt = request.query_params.get("format")
    width = int(request.query_params.get("width", "320"))
    height = int(request.query_params.get("height", "176"))

    # 1. Explicit multipart file upload
    if file is not None:
        raw = await file.read()
        return _decode_bytes_to_pil(raw, fmt=fmt, width=width, height=height)

    # 2. Check Content-Type for multipart, JSON, or raw binary image (JPEG/PNG/I420)
    ctype = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ctype:
        form = await request.form()
        if "file" in form and hasattr(form["file"], "read"):
            raw = await form["file"].read()  # type: ignore
            return _decode_bytes_to_pil(raw, fmt=fmt, width=width, height=height)
        if "sample" in form and not sample:
            sample = str(form["sample"])
        if "image_url" in form and not image_url:
            image_url = str(form["image_url"])
    elif "application/json" in ctype:
        try:
            body = await request.json()
            if isinstance(body, dict):
                if body.get("image_base64"):
                    b64_str = body["image_base64"]
                    if "," in b64_str:
                        b64_str = b64_str.split(",", 1)[1]
                    raw = base64.b64decode(b64_str)
                    return _decode_bytes_to_pil(
                        raw,
                        fmt=body.get("format") or fmt,
                        width=int(body.get("width", width)),
                        height=int(body.get("height", height)),
                    )
                if body.get("image_url") and not image_url:
                    image_url = str(body["image_url"])
                if body.get("sample") and not sample:
                    sample = str(body["sample"])
        except Exception:
            pass
    elif any(k in ctype for k in ("image/", "application/octet-stream", "video/raw")) or fmt == "i420":
        try:
            raw = await request.body()
        except Exception:
            raw = b""
        if raw:
            return _decode_bytes_to_pil(raw, fmt=fmt, width=width, height=height)
    else:
        # Also check if caller sent raw bytes without explicit content-type
        try:
            raw = await request.body()
        except Exception:
            raw = b""
        if raw and (len(raw) == (width * height * 3) // 2 or len(raw) == 84480 or raw[:2] == b"\xff\xd8" or raw[:4] == b"\x89PNG"):
            return _decode_bytes_to_pil(raw, fmt=fmt, width=width, height=height)

    # 3. Fetch from remote URL if provided
    if image_url:
        safe_url = _validate_remote_image_url(image_url)
        async with httpx.AsyncClient(timeout=10.0, verify=False, follow_redirects=False) as client:
            resp = await client.get(safe_url)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

    # 4. Built-in sample image fallback
    sample_key = (sample or "pose").lower()
    sample_path = SAMPLE_FILES.get(sample_key, SAMPLE_FILES["pose"])
    if sample_path.exists():
        return Image.open(sample_path).convert("RGB")

    raise HTTPException(status_code=400, detail="No valid image provided (upload file, pass image_url, image_base64, or sample=pose|face|parrot).")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    if DASHBOARD_FILE.exists():
        return HTMLResponse(DASHBOARD_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>TPUaaS Online</h1><p>Visit <a href='/docs'>/docs</a> for OpenAPI spec.</p>")


@app.get("/AGENT_GUIDE.md")
@app.get("/llms.txt")
async def serve_agent_guide() -> Response:
    if GUIDE_FILE.exists():
        return Response(content=GUIDE_FILE.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")
    return Response(content="# TPUaaS Agent Guide not found", media_type="text/markdown")


@app.get("/AGENT_BOARD.md")
@app.get("/v1/board")
async def serve_agent_board() -> Response:
    if BOARD_FILE.exists():
        return Response(content=BOARD_FILE.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")
    return Response(content="# TPUaaS Agent Board not found", media_type="text/markdown")


@app.post("/v1/board")
async def post_agent_board(request: Request) -> Dict[str, Any]:
    body = await request.json()
    agent = body.get("agent", "unknown-agent")
    project = body.get("project", "General")
    title = body.get("title", "Feature Request")
    mode = body.get("mode", "Mode 1 (Queued Cue)")
    budget = body.get("budget", "unspecified")
    details = body.get("details") or body.get("request") or ""
    import time as _t

    entry = (
        f"\n### [OPEN] {project} — {title} ({_t.strftime('%Y-%m-%d %H:%M:%S')})\n"
        f"- **Requesting Agent**: `{agent}`\n"
        f"- **Mode Needed**: `{mode}`\n"
        f"- **Target FPS / Latency Budget**: `{budget}`\n"
        f"- **Details**: {details}\n"
    )
    current = BOARD_FILE.read_text(encoding="utf-8") if BOARD_FILE.exists() else "# TPUaaS Agent Board\n"
    BOARD_FILE.write_text(current + entry, encoding="utf-8")
    return {"posted": True, "project": project, "title": title}


@app.get("/v1/streams")
async def list_streams() -> Dict[str, Any]:
    return {
        "dropped_stale_stream_frames": engine.dropped_stale_frames,
        "streams": engine.stream_mailboxes,
    }


@app.get("/v1/streams/{stream_id}")
async def get_stream_mailbox(stream_id: str) -> Dict[str, Any]:
    if stream_id not in engine.stream_mailboxes:
        raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' has no cached frame yet")
    return engine.stream_mailboxes[stream_id]


@app.get("/v1/health")
async def health() -> Dict[str, Any]:
    hw = engine.read_hardware_telemetry()
    return {
        "ok": hw["status"] == "ALIVE",
        "status": hw["status"],
        "temp_c": hw["temp_c"],
        "active_model": engine.active_model,
        "total_inferences": engine.total_inferences,
    }


@app.get("/v1/status")
async def status() -> Dict[str, Any]:
    st = engine.get_status()
    st["version"] = "1.3.0"
    st["voice"] = voice_engine.get_status()
    return st


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {
        "count": len(engine.model_stats),
        "active_sram_model": engine.active_model,
        "models": list(engine.model_stats.values()),
    }


@app.get("/v1/samples/{name}")
async def get_sample_image(name: str) -> Response:
    path = SAMPLE_FILES.get(name.lower())
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{name}' not found")
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.post("/v1/detect")
async def api_detect(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    model: str = Query(default="yolov9s", description="yolov9s | mobiledet_coco | efficientdet_coco | ssd_coco | ssd_face"),
    threshold: float = Query(default=0.25, ge=0.01, le=0.99),
    client: str = Query(default="default", description="ambiled | loona | drones | cameras | playground"),
    priority: str = Query(default="normal", description="realtime | high | normal | background"),
    stream_id: Optional[str] = Query(default=None, description="Set for Mode 2 (Latest-Frame-Wins Stream Mode)"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample, image_url=image_url)
    return engine.detect(
        img, model_name=model, threshold=threshold, client=client, priority=priority, stream_id=stream_id
    )


@app.post("/v1/pose")
async def api_pose(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    threshold: float = Query(default=0.20, ge=0.01, le=0.99),
    client: str = Query(default="loona"),
    priority: str = Query(default="realtime"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "pose", image_url=image_url)
    return engine.estimate_pose(img, threshold=threshold, client=client, priority=priority)


@app.post("/v1/segment")
async def api_segment(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    include_mask_b64: bool = Query(default=False),
    client: str = Query(default="ambiled"),
    priority: str = Query(default="normal"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "pose", image_url=image_url)
    return engine.segment(img, client=client, priority=priority, include_mask_b64=include_mask_b64)


@app.post("/v1/classify")
async def api_classify(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    top_k: int = Query(default=5, ge=1, le=20),
    threshold: float = Query(default=0.05, ge=0.0, le=0.99),
    client: str = Query(default="default"),
    priority: str = Query(default="normal"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "parrot", image_url=image_url)
    return engine.classify_image(img, top_k=top_k, threshold=threshold, client=client, priority=priority)


@app.post("/v1/embed")
async def api_embed(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    client: str = Query(default="loona"),
    priority: str = Query(default="realtime"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "face", image_url=image_url)
    return engine.extract_embedding(img, client=client, priority=priority)


@app.get("/v1/memory")
async def api_list_memory() -> Dict[str, Any]:
    items = [
        {k: v for k, v in item.items() if k != "embedding"}
        for item in engine.memory_bank
    ]
    return {"count": len(items), "items": items}


@app.post("/v1/memory/enroll")
async def api_enroll_memory(
    request: Request,
    name: str = Query(..., description="Name of person, pet, room, or object"),
    category: str = Query(default="person"),
    client: str = Query(default="loona"),
    file: Optional[UploadFile] = File(default=None),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "face", image_url=image_url)
    return engine.enroll_memory(img, name=name, category=category, client=client)


@app.post("/v1/memory/match")
async def api_match_memory(
    request: Request,
    category: Optional[str] = Query(default=None),
    top_k: int = Query(default=3, ge=1, le=10),
    threshold: float = Query(default=0.60, ge=0.0, le=1.0),
    client: str = Query(default="loona"),
    file: Optional[UploadFile] = File(default=None),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "face", image_url=image_url)
    return engine.match_memory(img, category=category, top_k=top_k, threshold=threshold, client=client)


@app.delete("/v1/memory/{item_id}")
async def api_delete_memory(item_id: str) -> Dict[str, Any]:
    before = len(engine.memory_bank)
    engine.memory_bank = [m for m in engine.memory_bank if m.get("id") != item_id]
    engine._save_memory_bank()
    return {"deleted": len(engine.memory_bank) < before, "remaining": len(engine.memory_bank)}


@app.post("/v1/annotate")
async def api_annotate(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    model: str = Query(default="yolov9s"),
    threshold: float = Query(default=0.25),
    include_pose: bool = Query(default=False),
    client: str = Query(default="playground"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Response:
    img = await resolve_input_image(request, file=file, sample=sample or "pose", image_url=image_url)
    det = engine.detect(img, model_name=model, threshold=threshold, client=client, priority="high")
    kps = None
    if include_pose:
        pose = engine.estimate_pose(img, threshold=0.2, client=client, priority="high")
        kps = pose["keypoints"]
    jpeg_bytes = engine.render_annotated_jpeg(img, detections=det["detections"], keypoints=kps)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


# --- Exclusive Take-Over Lease Endpoints (Default: Shared Architecture) ---

@app.get("/v1/lease")
async def api_get_lease() -> Dict[str, Any]:
    active = engine.get_active_lease()
    return {
        "architecture": "exclusive_takeover" if active else "shared_default",
        "exclusive_lease": active,
    }


@app.post("/v1/lease/acquire")
async def api_acquire_lease(
    request: Request,
    client: str = Query(default="drones", description="Client requesting exclusive take-over"),
    ttl_s: float = Query(default=30.0, ge=1.0, le=300.0, description="Auto-expiring lease TTL (seconds)"),
    idle_timeout_s: float = Query(default=10.0, ge=1.0, le=60.0, description="Auto-release if holder goes idle"),
    pinned_model: Optional[str] = Query(default="ssd_coco", description="Model to pre-warm and pin in EdgeTPU SRAM"),
    allow_same_model_sharing: bool = Query(
        default=True,
        description="Allow background tasks using the exact same SRAM-pinned model when queue_depth == 0",
    ),
) -> Dict[str, Any]:
    try:
        body = await request.json()
        if isinstance(body, dict):
            client = str(body.get("client", client))
            ttl_s = float(body.get("ttl_s", ttl_s))
            idle_timeout_s = float(body.get("idle_timeout_s", idle_timeout_s))
            if "pinned_model" in body:
                pinned_model = body["pinned_model"]
            elif "pin_model" in body:
                pinned_model = body["pin_model"]
            if "allow_same_model_sharing" in body:
                allow_same_model_sharing = bool(body["allow_same_model_sharing"])
    except Exception:
        pass
    return engine.acquire_lease(
        client=client,
        ttl_s=ttl_s,
        idle_timeout_s=idle_timeout_s,
        pin_model=pinned_model,
        allow_same_model_sharing=allow_same_model_sharing,
    )


@app.post("/v1/lease/release")
async def api_release_lease(
    request: Request,
    client: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    try:
        body = await request.json()
        if isinstance(body, dict) and "client" in body:
            client = str(body["client"])
    except Exception:
        pass
    return engine.release_lease(client=client)


# --- 4 Project-Specific Composite Pipelines + Lightshow + Downward Optical Flow ---

@app.post("/v1/projects/ambiled")
async def api_project_ambiled(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    include_segmentation: bool = Query(default=False),
    model: str = Query(default="ssd_coco", description="ssd_coco (default, 100% SRAM cache sharing) | yolov9s"),
    priority: str = Query(default="background", description="background (default, auto-ages after 85ms) | normal | high"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "pose", image_url=image_url)
    return engine.pipeline_ambiled(
        img, include_segmentation=include_segmentation, model_name=model, priority=priority
    )


@app.post("/v1/projects/loona")
async def api_project_loona(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    mode: str = Query(default="full", description="full (Cue Mode: Face+Pose+Objects) | fast (Driving Stream Mode: ~6ms ssd_coco)"),
    stream_id: Optional[str] = Query(default=None, description="Set for Mode 2 Latest-Frame-Wins"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "face", image_url=image_url)
    return engine.pipeline_loona(img, mode=mode, stream_id=stream_id)


@app.post("/v1/projects/drones")
async def api_project_drones(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    target_class: str = Query(default="person"),
    model: str = Query(default="ssd_coco", description="ssd_coco (default SRAM-locked ~5.9ms) or yolov9s"),
    stream_id: Optional[str] = Query(default=None, description="e.g. drone1_fwd, drone2_fwd, drone3_fwd"),
    camera_mount: str = Query(default="forward", description="forward | down"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "pose", image_url=image_url)
    return engine.pipeline_drones(
        img, target_class=target_class, model_name=model, stream_id=stream_id, camera_mount=camera_mount
    )


@app.post("/v1/projects/drones/optical_flow")
async def api_project_drones_optical_flow(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    stream_id: str = Query(default="drone1_down", description="e.g. drone1_down, drone2_down, drone3_down"),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "pose", image_url=image_url)
    return engine.pipeline_drones_optical_flow(img, stream_id=stream_id)


@app.get("/v1/projects/lightshow")
async def api_get_lightshow(
    safe_for_wheel_motion: bool = Query(default=True),
) -> Dict[str, Any]:
    return engine.get_lightshow_cue(safe_for_wheel_motion=safe_for_wheel_motion)


@app.post("/v1/projects/lightshow")
async def api_post_lightshow(request: Request) -> Dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected JSON object for lightshow state update")
    return engine.update_lightshow_state(body)


@app.post("/v1/projects/cameras")
async def api_project_cameras(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    camera_id: str = Query(default="front_porch"),
    threshold: float = Query(default=0.30),
    sample: Optional[str] = Query(default=None),
    image_url: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    img = await resolve_input_image(request, file=file, sample=sample or "pose", image_url=image_url)
    return engine.pipeline_cameras(img, camera_id=camera_id, threshold=threshold)


# --- Local Voice Pipeline & Loona Party Summon Endpoints (0% Coral TPU Load) ---

def _extract_pcm_from_wav_or_raw(raw_bytes: bytes) -> tuple[bytes, int]:
    """Extract S16_LE mono PCM and sample_rate from a WAV file, MCBV UDP packet, or raw S16_LE buffer."""
    if len(raw_bytes) > 16 and raw_bytes[:4] == b"MCBV":
        return raw_bytes[16:], 16000
    if len(raw_bytes) >= 44 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            if sw == 2 and ch == 1:
                return frames, sr
            if sw == 2 and ch == 2:
                import numpy as np
                stereo = np.frombuffer(frames, dtype=np.int16).reshape(-1, 2)
                mono = stereo.mean(axis=1).astype(np.int16)
                return mono.tobytes(), sr
        except Exception:
            pass
    return raw_bytes, 16000


@app.get("/v1/voice/status")
async def api_voice_status() -> Dict[str, Any]:
    return voice_engine.get_status()


@app.post("/v1/voice/utterance")
async def api_voice_utterance(
    request: Request,
    room: str = Query(default="living_room", description="Room slug, e.g. living_room"),
    text: Optional[str] = Query(default=None, description="Optional text transcript for direct HTTP testing"),
    require_wake_word: bool = Query(default=False, description="Require configured wake word in transcript"),
    dry_run: bool = Query(default=False, description="Publish intent to <topic_prefix>/voice/intent/dry_run instead of live bus"),
    file: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    # 1. Explicit query parameter text
    if text and text.strip():
        return voice_engine.handle_transcript(
            transcript=text.strip(),
            room=room,
            require_wake_word=require_wake_word,
            source="http_query",
            dry_run=dry_run,
        )

    # 2. Multipart file upload (.wav or raw .pcm)
    if file is not None:
        raw_audio = await file.read()
        pcm, sr = _extract_pcm_from_wav_or_raw(raw_audio)
        return voice_engine.process_pcm_utterance(
            raw_pcm=pcm,
            room=room,
            sample_rate=sr,
            require_wake_word=require_wake_word,
            source="http_upload",
            dry_run=dry_run,
        )

    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            body = await request.json()
            if isinstance(body, dict):
                room = str(body.get("room", room))
                if "require_wake_word" in body:
                    require_wake_word = bool(body["require_wake_word"])
                if "dry_run" in body:
                    dry_run = bool(body["dry_run"])
                txt_val = body.get("text") or body.get("utterance") or body.get("transcript") or body.get("command")
                if txt_val and str(txt_val).strip():
                    return voice_engine.handle_transcript(
                        transcript=str(txt_val).strip(),
                        room=room,
                        require_wake_word=require_wake_word,
                        source="http_json",
                        dry_run=dry_run,
                    )
                b64_audio = body.get("wav_base64") or body.get("pcm_base64") or body.get("audio_base64")
                if b64_audio:
                    raw_audio = base64.b64decode(str(b64_audio))
                    pcm, sr = _extract_pcm_from_wav_or_raw(raw_audio)
                    return voice_engine.process_pcm_utterance(
                        raw_pcm=pcm,
                        room=room,
                        sample_rate=int(body.get("sample_rate", sr)),
                        require_wake_word=require_wake_word,
                        source="http_json_audio",
                        dry_run=dry_run,
                    )
        except Exception:
            pass

    # 3. Raw binary WAV / S16_LE PCM / MCBV UDP packet body OR plain text body
    try:
        raw_body = await request.body()
    except Exception:
        raw_body = b""

    if raw_body:
        # Check if raw_body is plain UTF-8 text (and not binary WAV/MCBV/PCM)
        if raw_body[:4] not in (b"RIFF", b"MCBV") and len(raw_body) < 1024:
            try:
                decoded_txt = raw_body.decode("utf-8").strip()
                if decoded_txt and decoded_txt.isprintable():
                    if decoded_txt.startswith("{"):
                        data = json.loads(decoded_txt)
                        txt_val = data.get("text") or data.get("utterance") or data.get("transcript")
                        if txt_val:
                            return voice_engine.handle_transcript(
                                transcript=str(txt_val).strip(),
                                room=str(data.get("room", room)),
                                require_wake_word=bool(data.get("require_wake_word", require_wake_word)),
                                source="http_json",
                                dry_run=bool(data.get("dry_run", dry_run)),
                            )
                    return voice_engine.handle_transcript(
                        transcript=decoded_txt,
                        room=room,
                        require_wake_word=require_wake_word,
                        source="http_text",
                        dry_run=dry_run,
                    )
            except Exception:
                pass

        pcm, sr = _extract_pcm_from_wav_or_raw(raw_body)
        return voice_engine.process_pcm_utterance(
            raw_pcm=pcm,
            room=room,
            sample_rate=sr,
            require_wake_word=require_wake_word,
            source="http_binary_audio",
            dry_run=dry_run,
        )

    raise HTTPException(
        status_code=400,
        detail="Provide ?text=..., JSON {'text': ...}, or upload raw 16kHz S16_LE PCM / WAV audio.",
    )


@app.get("/v1/voice/tts/{filename}")
async def api_get_voice_tts(filename: str) -> Response:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "", filename)
    stem = safe_name[:-4] if safe_name.endswith(".wav") else safe_name
    candidates = [
        DATA_DIR / f"tts_{stem}.wav",
        DATA_DIR / safe_name,
        DATA_DIR / "tts_living_room.wav",
        DATA_DIR / "tts_latest.wav",
    ]
    for path in candidates:
        if path.is_file():
            return Response(content=path.read_bytes(), media_type="audio/wav")
    raise HTTPException(status_code=404, detail=f"TTS WAV '{filename}' not found")


@app.get("/v1/projects/loona/party_summon")
async def api_get_loona_party_summon() -> Dict[str, Any]:
    return {
        "loona_party_summon": voice_engine.loona_summon_state,
        "lightshow_cue": engine.get_lightshow_cue(safe_for_wheel_motion=True),
    }


@app.post("/v1/projects/loona/party_summon")
async def api_post_loona_party_summon(request: Request) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    try:
        raw = await request.json()
        if isinstance(raw, dict):
            body = raw
    except Exception:
        pass
    payload = {
        "action": body.get("action", "party_summon"),
        "target_room": body.get("target_room", "living_room"),
        "dance": body.get("dance", True),
        "scene_class": body.get("scene_class", "music_video"),
    }
    state = voice_engine.update_loona_summon(payload)
    mqtt_ok = mqtt_publish_one(
        voice_engine.mqtt_host,
        voice_engine.mqtt_port,
        voice_engine.mqtt_user,
        voice_engine.mqtt_pass,
        f"{TOPIC_PREFIX}/loona/command",
        json.dumps(payload).encode("utf-8"),
    )
    return {
        "loona_party_summon": state,
        "mqtt_published": mqtt_ok,
        "lightshow_cue": engine.get_lightshow_cue(safe_for_wheel_motion=True),
    }


@app.websocket("/ws/stream")
async def ws_stream(
    websocket: WebSocket,
    client: str = Query(default="drones"),
    model: str = Query(default="ssd_coco"),
    task: str = Query(default="detect", description="detect | pose | loona | drones | drones_optical_flow | ambiled | cameras"),
    mode: str = Query(default="fast", description="fast | full (for Loona)"),
    stream_id: Optional[str] = Query(default="ws_default"),
    threshold: float = Query(default=0.25),
    format: Optional[str] = Query(default=None),
    width: int = Query(default=320),
    height: int = Query(default=176),
) -> None:
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                img = _decode_bytes_to_pil(msg["bytes"], fmt=format, width=width, height=height)
            elif "text" in msg and msg["text"]:
                payload = json.loads(msg["text"])
                if payload.get("type") == "config":
                    client = payload.get("client", client)
                    model = payload.get("model", model)
                    task = payload.get("task", task)
                    mode = payload.get("mode", mode)
                    stream_id = payload.get("stream_id", stream_id)
                    threshold = float(payload.get("threshold", threshold))
                    format = payload.get("format", format)
                    width = int(payload.get("width", width))
                    height = int(payload.get("height", height))
                    await websocket.send_json(
                        {"ack": True, "client": client, "model": model, "task": task, "mode": mode, "stream_id": stream_id}
                    )
                    continue
                b64 = payload.get("image_base64", "")
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img = _decode_bytes_to_pil(base64.b64decode(b64), fmt=payload.get("format", format), width=width, height=height)
            else:
                continue

            if task == "pose":
                res = engine.estimate_pose(img, threshold=threshold, client=client, priority="realtime")
            elif task == "loona":
                res = engine.pipeline_loona(img, mode=mode, stream_id=stream_id)
            elif task == "drones":
                res = engine.pipeline_drones(img, model_name=model, stream_id=stream_id)
            elif task in ("drones_optical_flow", "optical_flow"):
                res = engine.pipeline_drones_optical_flow(img, stream_id=stream_id or "drone1_down")
            elif task == "ambiled":
                res = engine.pipeline_ambiled(img)
            elif task == "cameras":
                res = engine.pipeline_cameras(img, threshold=threshold)
            else:
                res = engine.detect(
                    img, model_name=model, threshold=threshold, client=client, priority="realtime", stream_id=stream_id
                )

            await websocket.send_json(res)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("TPUAAS_PORT", "8090")), log_level="info")
