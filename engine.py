from __future__ import annotations

import base64
import io
import json
import math
import os
import queue
import socket
import struct
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from pycoral.adapters import classify, common, detect
from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter


def _load_dotenv() -> None:
    """Load optional .env files from /opt/tpuaas/.env or package directory without external dependencies."""
    for env_path in (Path("/opt/tpuaas/.env"), Path(__file__).parent / ".env", Path(".env")):
        if env_path.is_file():
            try:
                for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            except Exception:
                pass


_load_dotenv()

PASCAL_VOC_LABELS = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tv",
}

MOVENET_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

PRIORITY_MAP = {
    "urgent": -1,
    "realtime": 0,
    "high": 1,
    "normal": 2,
    "background": 3,
}


def decode_i420_to_pil(raw: bytes, width: int = 320, height: int = 176) -> Image.Image:
    """Fast vectorized NumPy planar I420 (YUV420p) -> RGB PIL.Image decoder (<0.15ms on i3-N300)."""
    expected = width * height * 3 // 2
    if len(raw) < expected:
        raise ValueError(f"I420 buffer too short: got {len(raw)} bytes, expected {expected} for {width}x{height}")
    buf = np.frombuffer(raw, dtype=np.uint8, count=expected)
    y_size = width * height
    uv_w = width // 2
    uv_h = height // 2
    uv_size = uv_w * uv_h

    y = buf[:y_size].reshape((height, width)).astype(np.int32)
    u = buf[y_size : y_size + uv_size].reshape((uv_h, uv_w)).repeat(2, axis=0).repeat(2, axis=1).astype(np.int32) - 128
    v = buf[y_size + uv_size : y_size + 2 * uv_size].reshape((uv_h, uv_w)).repeat(2, axis=0).repeat(2, axis=1).astype(np.int32) - 128

    c = np.maximum(0, y - 16) * 298
    r = np.clip((c + 409 * v + 128) >> 8, 0, 255).astype(np.uint8)
    g = np.clip((c - 100 * u - 208 * v + 128) >> 8, 0, 255).astype(np.uint8)
    b = np.clip((c + 516 * u + 128) >> 8, 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def parse_label_file(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    ret: Dict[int, str] = {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            ret[int(parts[0])] = parts[1].strip()
        else:
            ret[idx] = line
    return ret


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _nms_boxes(
    boxes: List[Dict[str, Any]], iou_threshold: float = 0.45
) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in boxes:
        cx1, cy1, cx2, cy2 = cand["bbox"]
        c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
        suppress = False
        for k in kept:
            if k["class_id"] != cand["class_id"]:
                continue
            kx1, ky1, kx2, ky2 = k["bbox"]
            ix1 = max(cx1, kx1)
            iy1 = max(cy1, ky1)
            ix2 = min(cx2, kx2)
            iy2 = min(cy2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
                union = c_area + k_area - inter
                if union > 0 and (inter / union) > iou_threshold:
                    suppress = True
                    break
        if not suppress:
            kept.append(cand)
    return kept


class FastYOLOv9SepDecoder:
    """Vectorized YOLOv9 Separate-Output Decoder with pre-sigmoid logit filtering."""

    def __init__(self, img_size: Tuple[int, int] = (320, 320), reg_max: int = 16):
        self.img_w, self.img_h = img_size
        self.reg_max = reg_max
        self.dfl_weights = np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)
        self.strides = [8, 16, 32]
        anchors_list = []
        strides_list = []
        for s in self.strides:
            gh, gw = self.img_h // s, self.img_w // s
            sy, sx = np.meshgrid(
                np.arange(gh, dtype=np.float32) + 0.5,
                np.arange(gw, dtype=np.float32) + 0.5,
                indexing="ij",
            )
            anchors_list.append(np.stack((sx, sy), axis=-1).reshape(-1, 2))
            strides_list.append(np.full((gh * gw, 1), s, dtype=np.float32))
        self.anchors = np.concatenate(anchors_list, axis=0)  # (2100, 2)
        self.stride_tensor = np.concatenate(strides_list, axis=0)  # (2100, 1)

    def decode(
        self,
        raw_outputs: List[Tuple[np.ndarray, Dict[str, Any]]],
        threshold: float,
        labels: Dict[int, str],
        orig_w: int,
        orig_h: int,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        # Separate 64-channel box regression heads from class logit heads
        box_heads = []
        cls_heads = []
        for arr, detail in raw_outputs:
            scale, zp = detail["quantization"]
            deq = (arr[0].astype(np.float32) - zp) * scale
            if deq.shape[-1] == 64:
                box_heads.append(deq)
            else:
                cls_heads.append(deq)

        # Sort heads from largest spatial grid (1600 -> 400 -> 100)
        box_heads.sort(key=lambda x: -x.shape[0])
        cls_heads.sort(key=lambda x: -x.shape[0])

        cls_logits = np.concatenate(cls_heads, axis=0)  # (2100, num_classes)
        box_logits = np.concatenate(box_heads, axis=0)  # (2100, 64)

        # Fast logit pre-filter before computing sigmoid or DFL softmax
        clamped_thresh = min(max(threshold, 0.01), 0.99)
        logit_thresh = -math.log(1.0 / clamped_thresh - 1.0)

        max_logits = np.max(cls_logits, axis=1)
        keep_mask = max_logits > logit_thresh
        if not np.any(keep_mask):
            return []

        kept_indices = np.nonzero(keep_mask)[0]
        kept_cls_logits = cls_logits[kept_indices]  # (K, C)
        kept_box_logits = box_logits[kept_indices]  # (K, 64)
        kept_anchors = self.anchors[kept_indices]  # (K, 2)
        kept_strides = self.stride_tensor[kept_indices]  # (K, 1)

        # Compute DFL softmax only on surviving K anchors
        k_count = kept_box_logits.shape[0]
        dfl_reshaped = kept_box_logits.reshape(k_count, 4, self.reg_max).transpose(0, 2, 1)
        dfl_probs = _softmax(dfl_reshaped, axis=1)
        ltrb = np.sum(dfl_probs * self.dfl_weights, axis=1)  # (K, 4)

        x1y1 = (kept_anchors - ltrb[:, :2]) * kept_strides
        x2y2 = (kept_anchors + ltrb[:, 2:]) * kept_strides

        probs = 1.0 / (1.0 + np.exp(-kept_cls_logits))
        scale_x = orig_w / float(self.img_w)
        scale_y = orig_h / float(self.img_h)

        candidates: List[Dict[str, Any]] = []
        anchor_idx, class_idx = np.nonzero(probs >= threshold)
        for a_i, c_i in zip(anchor_idx, class_idx):
            score = float(probs[a_i, c_i])
            xmin = float(np.clip(x1y1[a_i, 0] * scale_x, 0, orig_w))
            ymin = float(np.clip(x1y1[a_i, 1] * scale_y, 0, orig_h))
            xmax = float(np.clip(x2y2[a_i, 0] * scale_x, 0, orig_w))
            ymax = float(np.clip(x2y2[a_i, 1] * scale_y, 0, orig_h))
            if xmax <= xmin or ymax <= ymin:
                continue
            cid = int(c_i)
            candidates.append(
                {
                    "class_id": cid,
                    "label": labels.get(cid, f"class_{cid}"),
                    "score": round(score, 4),
                    "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
                    "bbox_norm": [
                        round(xmin / orig_w, 4),
                        round(ymin / orig_h, 4),
                        round(xmax / orig_w, 4),
                        round(ymax / orig_h, 4),
                    ],
                }
            )

        return _nms_boxes(candidates, iou_threshold=iou_threshold)


@dataclass(order=True)
class ScheduledTask:
    priority: int
    locality_penalty: int
    seq: int
    model_name: str = field(compare=False)
    client: str = field(compare=False)
    fn: Callable[[], Any] = field(compare=False)
    future: Future = field(compare=False)
    enqueue_time: float = field(compare=False)
    stream_id: Optional[str] = field(default=None, compare=False)


class TPUEngine:
    def __init__(self, models_dir: Path, data_dir: Path):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.data_dir / "memory_bank.json"

        self.start_time = time.time()
        self.edge_tpus = list_edge_tpus()
        self.yolo_decoder = FastYOLOv9SepDecoder((320, 320))

        self.coco_labels = parse_label_file(self.models_dir / "coco_labels.txt")
        self.scrypted_labels = parse_label_file(self.models_dir / "scrypted_labels.txt")
        self.imagenet_labels = parse_label_file(self.models_dir / "imagenet_labels.txt")

        self.model_catalog: Dict[str, Dict[str, Any]] = {
            "yolov9s": {
                "file": "scrypted_yolov9s_relu_sep_320_edgetpu.tflite",
                "task": "detect",
                "description": "High-accuracy YOLOv9s 320x320 EdgeTPU security detector (person, vehicle, animal)",
                "labels": self.scrypted_labels,
                "recommended_for": ["cameras", "drones"],
            },
            "mobiledet_coco": {
                "file": "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
                "task": "detect",
                "description": "SSDLite MobileDet 320x320 90-class COCO detector",
                "labels": self.coco_labels,
                "recommended_for": ["ambiled", "loona", "drones", "cameras"],
            },
            "efficientdet_coco": {
                "file": "efficientdet_lite0_320_ptq_edgetpu.tflite",
                "task": "detect",
                "description": "EfficientDet-Lite0 320x320 90-class COCO detector",
                "labels": self.coco_labels,
                "recommended_for": ["cameras", "loona"],
            },
            "ssd_coco": {
                "file": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
                "task": "detect",
                "description": "Ultra-fast SSD MobileNet V2 300x300 90-class COCO detector",
                "labels": self.coco_labels,
                "recommended_for": ["ambiled", "drones", "loona"],
            },
            "ssd_face": {
                "file": "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
                "task": "face",
                "description": "Ultra-fast SSD MobileNet V2 320x320 human face detector",
                "labels": {0: "face"},
                "recommended_for": ["loona", "cameras"],
            },
            "movenet_pose": {
                "file": "movenet_single_pose_lightning_ptq_edgetpu.tflite",
                "task": "pose",
                "description": "MoveNet Single-Pose Lightning 192x192 17-keypoint body pose & gesture estimator",
                "labels": {},
                "recommended_for": ["loona", "drones", "cameras"],
            },
            "deeplab_seg": {
                "file": "deeplabv3_mnv2_pascal_quant_edgetpu.tflite",
                "task": "segment",
                "description": "DeepLabv3 MobileNetV2 513x513 21-class semantic segmentation (TV/screen, person, furniture)",
                "labels": PASCAL_VOC_LABELS,
                "recommended_for": ["ambiled", "loona"],
            },
            "mobilenet_cls": {
                "file": "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
                "task": "classify",
                "description": "MobileNet V2 224x224 1000-class ImageNet scene & object classifier",
                "labels": self.imagenet_labels,
                "recommended_for": ["ambiled", "loona", "cameras"],
            },
            "embeddings": {
                "file": "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite",
                "task": "embed",
                "description": "MobileNet V1 224x224 1024-D L2-normalized embedding extractor for visual memory & re-ID",
                "labels": {},
                "recommended_for": ["loona", "cameras", "drones"],
            },
        }

        self.interpreters: Dict[str, Any] = {}
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        self.client_stats: Dict[str, Dict[str, Any]] = {
            c: {"requests": 0, "total_ms": 0.0, "last_ms": 0.0, "last_model": None, "last_seen": None}
            for c in ["ambiled", "loona", "drones", "cameras", "playground", "default"]
        }

        self.active_model: Optional[str] = None
        self.total_inferences = 0
        self.sram_cache_hits = 0
        self.dropped_stale_frames = 0
        self.aged_promotions = 0
        self.stream_mailboxes: Dict[str, Dict[str, Any]] = {}
        self.recent_events: List[Dict[str, Any]] = []

        # Exclusive Take-Over Lease state (defaults to None = Shared Architecture)
        self.exclusive_lease: Optional[Dict[str, Any]] = None

        # CPU Optical Flow state per downward drone stream_id
        self._optflow_prev: Dict[str, Tuple[np.ndarray, float]] = {}

        # Cross-project AmbiLED <-> Loona Lightshow Sync state (0% TPU overhead)
        self.lightshow_state: Dict[str, Any] = {
            "lightshow_active": False,
            "slider_pct": 50,
            "scene_class": "auto",
            "is_scene_cut": False,
            "beat_pulse": False,
            "energy": 0.0,
            "palette_rgb": {
                "sky_west": [0, 0, 0],
                "sky_east": [0, 0, 0],
                "kitchen_wing": [0, 0, 0],
            },
            "ear_rgb": [0, 0, 0],
            "dance_action": "none",
            "mqtt_connected": False,
            "updated_at": 0.0,
        }

        self._seq = 0
        self._queue_lock = threading.Condition()
        self._pending_tasks: List[ScheduledTask] = []
        self._stop_event = threading.Event()

        self.prep_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="TPU-Prep")
        self.memory_bank: List[Dict[str, Any]] = self._load_memory_bank()

        # Start dedicated single-thread TPU hardware worker
        self.worker_thread = threading.Thread(target=self._tpu_worker_loop, name="TPU-Apex0-Worker", daemon=True)
        self.worker_thread.start()

        # Initialize and warm up all models on the TPU worker thread
        self._submit_sync("__init__", "system", "realtime", self._init_all_models)

        # Start optional background MQTT listener for AmbiLED -> Loona/Drone Lightshow Sync (if TPUAAS_MQTT_HOST is set)
        self.mqtt_thread = threading.Thread(target=self._mqtt_lightshow_listener, name="TPU-MQTT-Lightshow", daemon=True)
        self.mqtt_thread.start()

    def _load_memory_bank(self) -> List[Dict[str, Any]]:
        if self.memory_file.exists():
            try:
                return json.loads(self.memory_file.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_memory_bank(self) -> None:
        self.memory_file.write_text(json.dumps(self.memory_bank, indent=2), encoding="utf-8")

    def _init_all_models(self) -> None:
        for name, meta in self.model_catalog.items():
            path = self.models_dir / meta["file"]
            if not path.exists():
                continue
            interp = make_interpreter(str(path))
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]
            dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
            interp.set_tensor(inp["index"], dummy)
            t0 = time.perf_counter()
            interp.invoke()
            t1 = time.perf_counter()
            interp.invoke()
            warm_ms = (time.perf_counter() - t1) * 1000.0
            self.interpreters[name] = interp
            self.active_model = name
            self.model_stats[name] = {
                "name": name,
                "task": meta["task"],
                "description": meta["description"],
                "recommended_for": meta["recommended_for"],
                "input_shape": [int(x) for x in inp["shape"]],
                "input_dtype": inp["dtype"].__name__,
                "num_labels": len(meta["labels"]),
                "inferences": 0,
                "cache_hits": 0,
                "avg_tpu_ms": round(warm_ms, 2),
                "min_tpu_ms": round(warm_ms, 2),
                "last_tpu_ms": round(warm_ms, 2),
                "total_tpu_ms": 0.0,
            }

    def get_active_lease(self) -> Optional[Dict[str, Any]]:
        lease = self.exclusive_lease
        if not lease:
            return None
        now = time.time()
        if now >= lease["expires_at"] or (now - lease["last_activity"]) >= lease["idle_timeout_s"]:
            self.exclusive_lease = None
            return None
        return {
            **lease,
            "remaining_ttl_s": round(max(0.0, lease["expires_at"] - now), 1),
            "idle_s": round(now - lease["last_activity"], 1),
        }

    def acquire_lease(
        self,
        client: str,
        ttl_s: float = 30.0,
        idle_timeout_s: float = 10.0,
        pin_model: Optional[str] = "ssd_coco",
        allow_same_model_sharing: bool = True,
    ) -> Dict[str, Any]:
        ttl_s = min(max(ttl_s, 2.0), 300.0)
        idle_timeout_s = min(max(idle_timeout_s, 2.0), 60.0)
        now = time.time()
        self.exclusive_lease = {
            "owner": client,
            "pin_model": pin_model if (pin_model in self.interpreters) else None,
            "allow_same_model_sharing": bool(allow_same_model_sharing),
            "acquired_at": now,
            "expires_at": now + ttl_s,
            "idle_timeout_s": idle_timeout_s,
            "last_activity": now,
        }
        return {"acquired": True, "lease": self.get_active_lease()}

    def release_lease(self, client: Optional[str] = None) -> Dict[str, Any]:
        lease = self.get_active_lease()
        if not lease:
            return {"released": False, "reason": "no_active_lease"}
        if client and lease["owner"] != client and client != "admin":
            return {"released": False, "reason": f"owned_by_{lease['owner']}"}
        self.exclusive_lease = None
        return {"released": True}

    def _tpu_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._queue_lock:
                while not self._pending_tasks and not self._stop_event.is_set():
                    self._queue_lock.wait(timeout=0.5)
                if self._stop_event.is_set():
                    return
                now_perf = time.perf_counter()
                # Re-score locality penalty dynamically + apply Anti-Starvation Aging (>85ms wait -> priority -1)
                for task in self._pending_tasks:
                    task.locality_penalty = 0 if (task.model_name == self.active_model) else 1
                    if task.priority > 0 and (now_perf - task.enqueue_time) >= 0.085:
                        task.priority = -1
                        self.aged_promotions += 1
                self._pending_tasks.sort()
                task = self._pending_tasks.pop(0)

            try:
                result = task.fn()
                task.future.set_result(result)
            except Exception as exc:
                task.future.set_exception(exc)

    def _submit_sync(
        self,
        model_name: str,
        client: str,
        priority: str,
        fn: Callable[[], Any],
        stream_id: Optional[str] = None,
    ) -> Any:
        lease = self.get_active_lease()
        if lease and client != "system":
            if client == lease["owner"]:
                self.exclusive_lease["last_activity"] = time.time()  # type: ignore
                priority = "urgent"
            else:
                # Non-owner during Exclusive Take-Over: allow only if allow_same_model_sharing=True and model matches pin_model
                same_model = (lease.get("pin_model") is None) or (model_name == lease.get("pin_model"))
                if not (lease.get("allow_same_model_sharing", True) and same_model):
                    raise RuntimeError(
                        f"TPU_LOCKED_BY_EXCLUSIVE_LEASE: owned by '{lease['owner']}' (pinned={lease.get('pin_model')}, remaining={lease['remaining_ttl_s']}s)"
                    )
                priority = "background"

        prio_int = PRIORITY_MAP.get(priority, 2)
        fut: Future = Future()
        with self._queue_lock:
            if stream_id:
                for existing in self._pending_tasks:
                    if existing.stream_id == stream_id and existing.model_name == model_name:
                        # Mode 2 (Latest-Frame-Wins): overwrite stale queued frame in-place
                        existing.fn = fn
                        self.dropped_stale_frames += 1
                        fut = existing.future
                        break
                else:
                    self._seq += 1
                    locality = 0 if (model_name == self.active_model) else 1
                    task = ScheduledTask(
                        priority=prio_int,
                        locality_penalty=locality,
                        seq=self._seq,
                        model_name=model_name,
                        client=client,
                        fn=fn,
                        future=fut,
                        enqueue_time=time.perf_counter(),
                        stream_id=stream_id,
                    )
                    self._pending_tasks.append(task)
                    self._queue_lock.notify()
            else:
                self._seq += 1
                locality = 0 if (model_name == self.active_model) else 1
                task = ScheduledTask(
                    priority=prio_int,
                    locality_penalty=locality,
                    seq=self._seq,
                    model_name=model_name,
                    client=client,
                    fn=fn,
                    future=fut,
                    enqueue_time=time.perf_counter(),
                    stream_id=None,
                )
                self._pending_tasks.append(task)
                self._queue_lock.notify()
        return fut.result(timeout=30.0)

    def _record_metrics(
        self, model_name: str, client: str, tpu_ms: float, total_ms: float, cache_hit: bool, summary: str
    ) -> None:
        self.total_inferences += 1
        if cache_hit:
            self.sram_cache_hits += 1

        st = self.model_stats.get(model_name)
        if st:
            st["inferences"] += 1
            if cache_hit:
                st["cache_hits"] += 1
            st["total_tpu_ms"] += tpu_ms
            st["avg_tpu_ms"] = round(st["total_tpu_ms"] / st["inferences"], 2)
            st["min_tpu_ms"] = round(min(st["min_tpu_ms"], tpu_ms), 2)
            st["last_tpu_ms"] = round(tpu_ms, 2)

        c_key = client if client in self.client_stats else "default"
        cs = self.client_stats[c_key]
        cs["requests"] += 1
        cs["total_ms"] = round(cs["total_ms"] + total_ms, 2)
        cs["last_ms"] = round(total_ms, 2)
        cs["last_model"] = model_name
        cs["last_seen"] = time.strftime("%H:%M:%S")

        self.recent_events.append(
            {
                "ts": time.strftime("%H:%M:%S"),
                "client": client,
                "model": model_name,
                "tpu_ms": round(tpu_ms, 2),
                "total_ms": round(total_ms, 2),
                "cache_hit": cache_hit,
                "summary": summary,
            }
        )
        if len(self.recent_events) > 40:
            self.recent_events = self.recent_events[-40:]

    def read_hardware_telemetry(self) -> Dict[str, Any]:
        sysfs = Path("/sys/class/apex/apex_0")
        info: Dict[str, Any] = {
            "device": "/dev/apex_0",
            "type": "Google Coral Edge TPU (PCIe M.2)",
            "status": "ALIVE",
            "temp_c": None,
            "trip_point0_c": 84.8,
            "driver_version": "1.2",
            "pci_address": "0000:02:00.0",
        }
        if sysfs.exists():
            try:
                info["status"] = (sysfs / "status").read_text().strip()
                raw_temp = int((sysfs / "temp").read_text().strip())
                info["temp_c"] = round(raw_temp / 1000.0, 1)
                tp0 = int((sysfs / "trip_point0_temp").read_text().strip())
                info["trip_point0_c"] = round(tp0 / 1000.0, 1)
                info["pci_address"] = (sysfs / "pci_address").read_text().strip()
                info["driver_version"] = (sysfs / "driver_version").read_text().strip()
            except Exception:
                pass
        return info

    def get_status(self) -> Dict[str, Any]:
        hw = self.read_hardware_telemetry()
        hit_rate = (self.sram_cache_hits / self.total_inferences * 100.0) if self.total_inferences > 0 else 100.0
        active_lease = self.get_active_lease()
        return {
            "service": "TPUaaS (Coral EdgeTPU-as-a-Service)",
            "version": "1.2.0",
            "uptime_s": round(time.time() - self.start_time, 1),
            "hardware": hw,
            "scheduler": {
                "architecture": "exclusive_takeover" if active_lease else "shared_default",
                "exclusive_lease": active_lease,
                "modes": [
                    "queue (Mode 1: FIFO + Anti-Starvation 85ms Aging + SRAM Locality)",
                    "stream (Mode 2: Latest-Frame-Wins Mailbox)",
                    "cpu_optical_flow (0% TPU i3-N300 Downward Drone Belly Cam)",
                ],
                "active_sram_model": self.active_model,
                "queue_depth": len(self._pending_tasks),
                "total_inferences": self.total_inferences,
                "sram_cache_hits": self.sram_cache_hits,
                "sram_cache_hit_rate_pct": round(hit_rate, 1),
                "dropped_stale_stream_frames": self.dropped_stale_frames,
                "aged_mode1_promotions": self.aged_promotions,
                "active_streams": list(self.stream_mailboxes.keys()),
                "memory_bank_items": len(self.memory_bank),
            },
            "lightshow": self.lightshow_state,
            "clients": self.client_stats,
            "models": list(self.model_stats.values()),
            "recent_events": list(reversed(self.recent_events[-15:])),
        }

    def detect(
        self,
        img: Image.Image,
        model_name: str = "yolov9s",
        threshold: float = 0.25,
        client: str = "default",
        priority: str = "normal",
        stream_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        if model_name not in self.interpreters:
            model_name = "yolov9s" if "yolov9s" in self.interpreters else next(iter(self.interpreters))

        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size
        interp = self.interpreters[model_name]
        inp_detail = interp.get_input_details()[0]
        _, in_h, in_w, _ = inp_detail["shape"]
        resized = rgb.resize((int(in_w), int(in_h)), Image.Resampling.BILINEAR)

        is_yolov9 = model_name == "yolov9s"
        if is_yolov9:
            arr = np.asarray(resized, dtype=np.uint8)
            inp_tensor = (arr.astype(np.int16) - 128).astype(np.int8)[None, ...]

            def _run_yolo() -> Tuple[List[Tuple[np.ndarray, Dict[str, Any]]], float, bool]:
                cache_hit = self.active_model == model_name
                t0 = time.perf_counter()
                interp.set_tensor(inp_detail["index"], inp_tensor)
                interp.invoke()
                out_details = interp.get_output_details()
                raw_outs = [(interp.get_tensor(d["index"]).copy(), d) for d in out_details]
                tpu_ms = (time.perf_counter() - t0) * 1000.0
                self.active_model = model_name
                return raw_outs, tpu_ms, cache_hit

            raw_outs, tpu_ms, cache_hit = self._submit_sync(
                model_name, client, priority, _run_yolo, stream_id=stream_id
            )
            detections = self.yolo_decoder.decode(
                raw_outs,
                threshold=threshold,
                labels=self.model_catalog[model_name]["labels"],
                orig_w=orig_w,
                orig_h=orig_h,
            )
        else:
            scale_x = orig_w / float(in_w)
            scale_y = orig_h / float(in_h)
            labels = self.model_catalog[model_name]["labels"]

            def _run_ssd() -> Tuple[List[Any], float, bool]:
                cache_hit = self.active_model == model_name
                t0 = time.perf_counter()
                common.set_input(interp, resized)
                interp.invoke()
                objs = detect.get_objects(interp, score_threshold=threshold, image_scale=(1.0, 1.0))
                tpu_ms = (time.perf_counter() - t0) * 1000.0
                self.active_model = model_name
                return objs, tpu_ms, cache_hit

            objs, tpu_ms, cache_hit = self._submit_sync(
                model_name, client, priority, _run_ssd, stream_id=stream_id
            )
            detections = []
            for obj in objs:
                xmin = float(np.clip(obj.bbox.xmin * scale_x, 0, orig_w))
                ymin = float(np.clip(obj.bbox.ymin * scale_y, 0, orig_h))
                xmax = float(np.clip(obj.bbox.xmax * scale_x, 0, orig_w))
                ymax = float(np.clip(obj.bbox.ymax * scale_y, 0, orig_h))
                cid = int(obj.id)
                detections.append(
                    {
                        "class_id": cid,
                        "label": labels.get(cid, f"class_{cid}"),
                        "score": round(float(obj.score), 4),
                        "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
                        "bbox_norm": [
                            round(xmin / orig_w, 4),
                            round(ymin / orig_h, 4),
                            round(xmax / orig_w, 4),
                            round(ymax / orig_h, 4),
                        ],
                    }
                )
            detections.sort(key=lambda d: d["score"], reverse=True)

        total_ms = (time.perf_counter() - t_start) * 1000.0
        top_summary = (
            f"{len(detections)} objs ({', '.join(d['label'] for d in detections[:3])})"
            if detections
            else "0 objs"
        )
        self._record_metrics(model_name, client, tpu_ms, total_ms, cache_hit, top_summary)

        res = {
            "model": model_name,
            "client": client,
            "stream_id": stream_id,
            "mode": "stream" if stream_id else "queue",
            "image_size": [orig_w, orig_h],
            "tpu_ms": round(tpu_ms, 2),
            "total_ms": round(total_ms, 2),
            "sram_cache_hit": cache_hit,
            "count": len(detections),
            "detections": detections,
        }
        if stream_id:
            self.stream_mailboxes[stream_id] = {**res, "updated_at": time.time()}
        return res

    def estimate_pose(
        self,
        img: Image.Image,
        threshold: float = 0.2,
        client: str = "loona",
        priority: str = "realtime",
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        model_name = "movenet_pose"
        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size
        interp = self.interpreters[model_name]
        inp_detail = interp.get_input_details()[0]
        _, in_h, in_w, _ = inp_detail["shape"]
        resized = rgb.resize((int(in_w), int(in_h)), Image.Resampling.BILINEAR)

        def _run_pose() -> Tuple[np.ndarray, float, bool]:
            cache_hit = self.active_model == model_name
            t0 = time.perf_counter()
            common.set_input(interp, resized)
            interp.invoke()
            out = common.output_tensor(interp, 0).copy()  # (1, 1, 17, 3) -> [y, x, score]
            tpu_ms = (time.perf_counter() - t0) * 1000.0
            self.active_model = model_name
            return out, tpu_ms, cache_hit

        raw_kps, tpu_ms, cache_hit = self._submit_sync(model_name, client, priority, _run_pose)
        kps_arr = raw_kps.reshape(17, 3)

        keypoints: Dict[str, Dict[str, float]] = {}
        kp_list: List[Dict[str, Any]] = []
        for idx, name in enumerate(MOVENET_KEYPOINT_NAMES):
            ky, kx, kscore = float(kps_arr[idx, 0]), float(kps_arr[idx, 1]), float(kps_arr[idx, 2])
            entry = {
                "name": name,
                "x": round(kx * orig_w, 1),
                "y": round(ky * orig_h, 1),
                "x_norm": round(kx, 4),
                "y_norm": round(ky, 4),
                "score": round(kscore, 4),
                "visible": kscore >= threshold,
            }
            keypoints[name] = entry
            kp_list.append(entry)

        # Derive high-level human gesture for Loona / Drones / Camera automation
        gesture = "none"
        ls = keypoints["left_shoulder"]
        rs = keypoints["right_shoulder"]
        lw = keypoints["left_wrist"]
        rw = keypoints["right_wrist"]
        lh = keypoints["left_hip"]
        rh = keypoints["right_hip"]
        lk = keypoints["left_knee"]
        rk = keypoints["right_knee"]

        l_raised = lw["visible"] and ls["visible"] and (lw["y_norm"] < ls["y_norm"] - 0.05)
        r_raised = rw["visible"] and rs["visible"] and (rw["y_norm"] < rs["y_norm"] - 0.05)
        if l_raised and r_raised:
            gesture = "hands_up"
        elif l_raised:
            gesture = "waving_left"
        elif r_raised:
            gesture = "waving_right"
        elif (
            lh["visible"]
            and lk["visible"]
            and abs(lh["y_norm"] - lk["y_norm"]) < 0.12
        ):
            gesture = "crouching_or_seated"
        elif ls["visible"] and rs["visible"]:
            gesture = "standing_attentive"

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self._record_metrics(model_name, client, tpu_ms, total_ms, cache_hit, f"pose ({gesture})")

        return {
            "model": model_name,
            "client": client,
            "image_size": [orig_w, orig_h],
            "tpu_ms": round(tpu_ms, 2),
            "total_ms": round(total_ms, 2),
            "sram_cache_hit": cache_hit,
            "gesture": gesture,
            "keypoints": kp_list,
            "keypoints_by_name": keypoints,
        }

    def segment(
        self,
        img: Image.Image,
        client: str = "ambiled",
        priority: str = "normal",
        include_mask_b64: bool = False,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        model_name = "deeplab_seg"
        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size
        interp = self.interpreters[model_name]
        inp_detail = interp.get_input_details()[0]
        _, in_h, in_w, _ = inp_detail["shape"]
        resized = rgb.resize((int(in_w), int(in_h)), Image.Resampling.BILINEAR)

        def _run_seg() -> Tuple[np.ndarray, float, bool]:
            cache_hit = self.active_model == model_name
            t0 = time.perf_counter()
            common.set_input(interp, resized)
            interp.invoke()
            mask = common.output_tensor(interp, 0)[0].copy()  # (513, 513)
            tpu_ms = (time.perf_counter() - t0) * 1000.0
            self.active_model = model_name
            return mask, tpu_ms, cache_hit

        mask, tpu_ms, cache_hit = self._submit_sync(model_name, client, priority, _run_seg)
        total_px = float(mask.size)
        unique_ids, counts = np.unique(mask, return_counts=True)

        regions: List[Dict[str, Any]] = []
        scale_x = orig_w / float(in_w)
        scale_y = orig_h / float(in_h)
        for cid_raw, cnt in zip(unique_ids, counts):
            cid = int(cid_raw)
            label = PASCAL_VOC_LABELS.get(cid, f"class_{cid}")
            coverage = float(cnt) / total_px
            if cid == 0 or coverage < 0.002:
                continue
            ys, xs = np.nonzero(mask == cid)
            xmin = float(np.min(xs) * scale_x)
            ymin = float(np.min(ys) * scale_y)
            xmax = float(np.max(xs) * scale_x)
            ymax = float(np.max(ys) * scale_y)
            regions.append(
                {
                    "class_id": cid,
                    "label": label,
                    "coverage_pct": round(coverage * 100.0, 2),
                    "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
                    "bbox_norm": [
                        round(xmin / orig_w, 4),
                        round(ymin / orig_h, 4),
                        round(xmax / orig_w, 4),
                        round(ymax / orig_h, 4),
                    ],
                }
            )
        regions.sort(key=lambda r: r["coverage_pct"], reverse=True)

        mask_b64 = None
        if include_mask_b64:
            # Downsample mask to 160x90 PNG for lightweight network transport
            mask_img = Image.fromarray(mask.astype(np.uint8), mode="L").resize(
                (160, 90), Image.Resampling.NEAREST
            )
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        total_ms = (time.perf_counter() - t_start) * 1000.0
        summary = f"{len(regions)} regions ({', '.join(r['label'] for r in regions[:3])})"
        self._record_metrics(model_name, client, tpu_ms, total_ms, cache_hit, summary)

        return {
            "model": model_name,
            "client": client,
            "image_size": [orig_w, orig_h],
            "tpu_ms": round(tpu_ms, 2),
            "total_ms": round(total_ms, 2),
            "sram_cache_hit": cache_hit,
            "regions": regions,
            "mask_png_b64": mask_b64,
        }

    def classify_image(
        self,
        img: Image.Image,
        top_k: int = 5,
        threshold: float = 0.05,
        client: str = "default",
        priority: str = "normal",
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        model_name = "mobilenet_cls"
        rgb = img.convert("RGB")
        interp = self.interpreters[model_name]
        inp_detail = interp.get_input_details()[0]
        _, in_h, in_w, _ = inp_detail["shape"]
        resized = rgb.resize((int(in_w), int(in_h)), Image.Resampling.BILINEAR)

        def _run_cls() -> Tuple[List[Any], float, bool]:
            cache_hit = self.active_model == model_name
            t0 = time.perf_counter()
            common.set_input(interp, resized)
            interp.invoke()
            classes = classify.get_classes(interp, top_k=top_k, score_threshold=threshold)
            tpu_ms = (time.perf_counter() - t0) * 1000.0
            self.active_model = model_name
            return classes, tpu_ms, cache_hit

        classes, tpu_ms, cache_hit = self._submit_sync(model_name, client, priority, _run_cls)
        labels = self.model_catalog[model_name]["labels"]
        results = [
            {
                "class_id": int(c.id),
                "label": labels.get(int(c.id), f"class_{c.id}"),
                "score": round(float(c.score), 4),
            }
            for c in classes
        ]
        total_ms = (time.perf_counter() - t_start) * 1000.0
        top_label = results[0]["label"] if results else "none"
        self._record_metrics(model_name, client, tpu_ms, total_ms, cache_hit, f"cls ({top_label})")

        return {
            "model": model_name,
            "client": client,
            "tpu_ms": round(tpu_ms, 2),
            "total_ms": round(total_ms, 2),
            "sram_cache_hit": cache_hit,
            "predictions": results,
        }

    def extract_embedding(
        self,
        img: Image.Image,
        client: str = "loona",
        priority: str = "realtime",
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        model_name = "embeddings"
        rgb = img.convert("RGB")
        interp = self.interpreters[model_name]
        inp_detail = interp.get_input_details()[0]
        _, in_h, in_w, _ = inp_detail["shape"]
        resized = rgb.resize((int(in_w), int(in_h)), Image.Resampling.BILINEAR)

        def _run_emb() -> Tuple[np.ndarray, Dict[str, Any], float, bool]:
            cache_hit = self.active_model == model_name
            t0 = time.perf_counter()
            common.set_input(interp, resized)
            interp.invoke()
            out_detail = interp.get_output_details()[0]
            raw = interp.get_tensor(out_detail["index"]).copy()
            tpu_ms = (time.perf_counter() - t0) * 1000.0
            self.active_model = model_name
            return raw, out_detail, tpu_ms, cache_hit

        raw, out_detail, tpu_ms, cache_hit = self._submit_sync(model_name, client, priority, _run_emb)
        scale, zp = out_detail["quantization"]
        vec = (raw.flatten().astype(np.float32) - zp) * (scale if scale > 0 else 1.0)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self._record_metrics(model_name, client, tpu_ms, total_ms, cache_hit, "1024-D embedding")

        return {
            "model": model_name,
            "client": client,
            "dimension": int(vec.size),
            "tpu_ms": round(tpu_ms, 2),
            "total_ms": round(total_ms, 2),
            "sram_cache_hit": cache_hit,
            "embedding": [round(float(x), 5) for x in vec],
        }

    def enroll_memory(
        self,
        img: Image.Image,
        name: str,
        category: str = "person",
        metadata: Optional[Dict[str, Any]] = None,
        client: str = "loona",
    ) -> Dict[str, Any]:
        emb_res = self.extract_embedding(img, client=client, priority="high")
        item = {
            "id": str(uuid.uuid4())[:8],
            "name": name,
            "category": category,
            "metadata": metadata or {},
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding": emb_res["embedding"],
        }
        self.memory_bank.append(item)
        self._save_memory_bank()
        return {
            "enrolled": True,
            "id": item["id"],
            "name": name,
            "category": category,
            "total_memories": len(self.memory_bank),
            "tpu_ms": emb_res["tpu_ms"],
        }

    def match_memory(
        self,
        img: Image.Image,
        category: Optional[str] = None,
        top_k: int = 3,
        threshold: float = 0.65,
        client: str = "loona",
    ) -> Dict[str, Any]:
        emb_res = self.extract_embedding(img, client=client, priority="realtime")
        q_vec = np.asarray(emb_res["embedding"], dtype=np.float32)

        matches: List[Dict[str, Any]] = []
        for item in self.memory_bank:
            if category and item.get("category") != category:
                continue
            m_vec = np.asarray(item["embedding"], dtype=np.float32)
            sim = float(np.dot(q_vec, m_vec))
            if sim >= threshold:
                matches.append(
                    {
                        "id": item["id"],
                        "name": item["name"],
                        "category": item.get("category", "general"),
                        "similarity": round(sim, 4),
                        "metadata": item.get("metadata", {}),
                    }
                )
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return {
            "client": client,
            "tpu_ms": emb_res["tpu_ms"],
            "total_ms": emb_res["total_ms"],
            "matches": matches[:top_k],
            "best_match": matches[0] if matches else None,
        }

    # --- Cross-Project AmbiLED <-> Loona Lightshow Sync (0% TPU Usage) ---

    def _compute_dance_action(self, safe_for_wheels: bool = True) -> str:
        ls = self.lightshow_state
        if not ls.get("lightshow_active"):
            return "none"
        energy = float(ls.get("energy", 0.0))
        beat = bool(ls.get("beat_pulse") or ls.get("is_scene_cut"))
        sclass = ls.get("scene_class", "auto")

        if beat and energy >= 0.72 and sclass == "music_video":
            return "hype_jump" if safe_for_wheels else "ear_groove"
        if beat and energy >= 0.45:
            return "beat_spin" if safe_for_wheels else "head_bob"
        if energy >= 0.55 or sclass == "music_video":
            return "ear_groove"
        if energy >= 0.25:
            return "head_bob"
        return "gentle_sway"

    def _compute_swarm_choreography(self, drone_id: int = 1, safe_to_dance: bool = True) -> Dict[str, Any]:
        """Compute synchronized 3-drone + Loona music-video swarm choreography (0% TPU usage).
        Automatically zeroes all stick deltas if safe_to_dance is False (obstacle safety interlock).
        """
        ls = self.lightshow_state
        pal = ls.get("palette_rgb", {})
        if drone_id == 1:
            zone = "kitchen_wing"
            led_rgb = pal.get("kitchen_wing", [0, 0, 0])
        elif drone_id == 2:
            zone = "sky_west"
            led_rgb = pal.get("sky_west", [0, 0, 0])
        else:
            zone = "sky_east"
            led_rgb = pal.get("sky_east", [0, 0, 0])

        if not ls.get("lightshow_active"):
            return {
                "drone_id": drone_id,
                "assigned_zone": zone,
                "safe_to_dance": safe_to_dance,
                "formation_move": "idle",
                "led_rgb": led_rgb,
                "dance_roll_delta": 0.0,
                "dance_pitch_delta": 0.0,
                "dance_throttle_delta": 0.0,
                "dance_yaw_delta": 0.0,
            }

        if not safe_to_dance:
            return {
                "drone_id": drone_id,
                "assigned_zone": zone,
                "safe_to_dance": False,
                "formation_move": "safety_hold_obstacle",
                "led_rgb": led_rgb,
                "dance_roll_delta": 0.0,
                "dance_pitch_delta": 0.0,
                "dance_throttle_delta": 0.0,
                "dance_yaw_delta": 0.0,
            }

        energy = float(ls.get("energy", 0.0))
        beat = bool(ls.get("beat_pulse") or ls.get("is_scene_cut"))
        sclass = ls.get("scene_class", "auto")
        # 120-degree phase offset across the 3-drone fleet (2 Hz base groove frequency)
        phase = time.time() * 2.0 * math.pi * 0.75 + (drone_id - 1) * (2.0 * math.pi / 3.0)
        amp = min(0.12, max(0.03, energy * 0.14))

        if beat and energy >= 0.72 and sclass == "music_video":
            move = "beat_drop_pirouette"
            roll_d = 0.0
            pitch_d = 0.0
            thr_d = round(amp * 1.1, 4)
            yaw_d = round(0.28 if drone_id != 2 else -0.28, 4)
        elif energy >= 0.50 or sclass == "music_video":
            move = "synchronized_figure8"
            roll_d = round(math.sin(phase) * amp, 4)
            pitch_d = round(math.sin(2.0 * phase) * (amp * 0.65), 4)
            thr_d = round(math.cos(phase) * (amp * 0.5), 4)
            yaw_d = 0.0
        else:
            move = "west_east_wave"
            roll_d = round(math.sin(phase) * (amp * 0.7), 4)
            pitch_d = 0.0
            thr_d = round(math.sin(phase) * (amp * 0.8), 4)
            yaw_d = 0.0

        return {
            "drone_id": drone_id,
            "assigned_zone": zone,
            "safe_to_dance": True,
            "formation_move": move,
            "led_rgb": led_rgb,
            "dance_roll_delta": roll_d,
            "dance_pitch_delta": pitch_d,
            "dance_throttle_delta": thr_d,
            "dance_yaw_delta": yaw_d,
        }

    def update_lightshow_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ls = self.lightshow_state
        if "lightshow_active" in payload:
            ls["lightshow_active"] = bool(payload["lightshow_active"])
        if "slider_pct" in payload:
            ls["slider_pct"] = int(payload["slider_pct"])
        if "scene_class" in payload:
            ls["scene_class"] = str(payload["scene_class"])
        if "is_scene_cut" in payload:
            ls["is_scene_cut"] = bool(payload["is_scene_cut"])
        if "beat_pulse" in payload:
            ls["beat_pulse"] = bool(payload["beat_pulse"])
        if "energy" in payload:
            ls["energy"] = round(float(payload["energy"]), 3)
        if "palette_rgb" in payload and isinstance(payload["palette_rgb"], dict):
            ls["palette_rgb"].update(payload["palette_rgb"])

        # Support direct diamond MQTT schema (up_rgb, down_rgb, left_rgb/cabinet_rgb, up_bri, down_bri, left_bri/cabinet_bri)
        if any(k in payload for k in ("up_rgb", "down_rgb", "left_rgb", "cabinet_rgb")):
            up_rgb = payload.get("up_rgb", ls["palette_rgb"].get("sky_west", [0, 0, 0]))
            down_rgb = payload.get("down_rgb", ls["palette_rgb"].get("sky_east", [0, 0, 0]))
            left_rgb = payload.get("cabinet_rgb") or payload.get("left_rgb") or ls["palette_rgb"].get("kitchen_wing", [0, 0, 0])
            prev_west = np.asarray(ls["palette_rgb"].get("sky_west", [0, 0, 0]), dtype=np.float32)
            new_west = np.asarray(up_rgb, dtype=np.float32)
            delta = float(np.mean(np.abs(new_west - prev_west))) / 255.0
            max_bri = max(
                int(payload.get("up_bri", 0)),
                int(payload.get("down_bri", 0)),
                int(payload.get("left_bri", 0)),
                int(payload.get("cabinet_bri", 0)),
            )
            if "lightshow_active" not in payload:
                ls["lightshow_active"] = max_bri > 5
            ls["palette_rgb"] = {
                "sky_west": [int(x) for x in up_rgb[:3]],
                "sky_east": [int(x) for x in down_rgb[:3]],
                "kitchen_wing": [int(x) for x in left_rgb[:3]],
            }
            if "energy" not in payload:
                ls["energy"] = round(min(1.0, delta * 2.4 + (max_bri / 255.0) * 0.35), 3)
            if "beat_pulse" not in payload:
                ls["beat_pulse"] = delta >= 0.22

        # Derive Loona ear_rgb from dominant vivid room color
        pal = ls["palette_rgb"]
        west = pal.get("sky_west", [0, 0, 0])
        east = pal.get("sky_east", [0, 0, 0])
        ls["ear_rgb"] = west if sum(west) >= sum(east) else east
        ls["dance_action"] = self._compute_dance_action(safe_for_wheels=True)
        ls["updated_at"] = round(time.time(), 2)
        self.stream_mailboxes["lightshow_cue"] = dict(ls)
        return dict(ls)

    def get_lightshow_cue(self, safe_for_wheel_motion: bool = True) -> Dict[str, Any]:
        cue = dict(self.lightshow_state)
        cue["safe_for_wheel_motion"] = safe_for_wheel_motion
        cue["dance_action"] = self._compute_dance_action(safe_for_wheels=safe_for_wheel_motion)
        cue["drone_swarm"] = [
            self._compute_swarm_choreography(drone_id=i, safe_to_dance=safe_for_wheel_motion)
            for i in (1, 2, 3)
        ]
        return cue

    def _mqtt_lightshow_listener(self) -> None:
        """Optional pure-stdlib MQTT 3.1.1 subscriber for live AmbiLED/RPMLight telemetry.
        Enabled when TPUAAS_MQTT_HOST is configured in the environment or .env file.
        """
        mqtt_host = os.environ.get("TPUAAS_MQTT_HOST", "").strip()
        if not mqtt_host:
            return
        mqtt_port = int(os.environ.get("TPUAAS_MQTT_PORT", "1883"))
        mqtt_user_str = os.environ.get("TPUAAS_MQTT_USER", "").strip()
        mqtt_pass_str = os.environ.get("TPUAAS_MQTT_PASS", "")
        topic_prefix = os.environ.get("TPUAAS_TOPIC_PREFIX", "tpuaas").strip("/") or "tpuaas"
        loona_cmd_topic = f"{topic_prefix}/loona/command"
        topics_raw = os.environ.get("TPUAAS_MQTT_TOPICS", f"rpmlight/#,{loona_cmd_topic}")
        topic_filters = [t.strip().encode("utf-8") for t in topics_raw.split(",") if t.strip()]
        if loona_cmd_topic.encode("utf-8") not in topic_filters:
            topic_filters.append(loona_cmd_topic.encode("utf-8"))

        def _enc_str(b: bytes) -> bytes:
            return struct.pack("!H", len(b)) + b

        def _enc_varlen(length: int) -> bytes:
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

        while not self._stop_event.is_set():
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(30.0)
                sock.connect((mqtt_host, mqtt_port))

                # Build MQTT 3.1.1 CONNECT packet (Clean Session + optional Username/Password)
                client_id = f"tpuaas-{uuid.uuid4().hex[:6]}".encode("utf-8")
                if mqtt_user_str:
                    var_hdr = b"\x00\x04MQTT\x04\xc2\x00\x3c"  # flags=0xC2 (user+pass+clean), keepalive=60s
                    payload = (
                        _enc_str(client_id)
                        + _enc_str(mqtt_user_str.encode("utf-8"))
                        + _enc_str(mqtt_pass_str.encode("utf-8"))
                    )
                else:
                    var_hdr = b"\x00\x04MQTT\x04\x02\x00\x3c"  # flags=0x02 (clean session), keepalive=60s
                    payload = _enc_str(client_id)
                rem = var_hdr + payload
                sock.sendall(b"\x10" + _enc_varlen(len(rem)) + rem)
                connack = sock.recv(4)
                if len(connack) < 4 or connack[3] != 0:
                    time.sleep(10.0)
                    continue

                # Send SUBSCRIBE packet (Packet ID = 1, QoS = 0 for all configured topics)
                sub_rem = b"\x00\x01" + b"".join(_enc_str(tf) + b"\x00" for tf in topic_filters)
                sock.sendall(b"\x82" + _enc_varlen(len(sub_rem)) + sub_rem)
                self.lightshow_state["mqtt_connected"] = True

                last_ping = time.time()
                buf = b""
                while not self._stop_event.is_set():
                    if time.time() - last_ping >= 25.0:
                        sock.sendall(b"\xc0\x00")  # PINGREQ
                        last_ping = time.time()
                    try:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                    except socket.timeout:
                        continue

                    while len(buf) >= 2:
                        pkt_type = buf[0]
                        mult = 1
                        rem_len = 0
                        idx = 1
                        while idx < len(buf):
                            b_val = buf[idx]
                            rem_len += (b_val & 0x7F) * mult
                            mult *= 128
                            idx += 1
                            if (b_val & 0x80) == 0:
                                break
                        else:
                            break
                        if len(buf) < idx + rem_len:
                            break
                        pkt_body = buf[idx : idx + rem_len]
                        buf = buf[idx + rem_len :]

                        if (pkt_type & 0xF0) == 0x30 and len(pkt_body) >= 2:  # PUBLISH
                            tlen = struct.unpack("!H", pkt_body[:2])[0]
                            topic_str = pkt_body[2 : 2 + tlen].decode("utf-8", errors="ignore")
                            pos = 2 + tlen
                            if ((pkt_type >> 1) & 0x03) > 0:
                                pos += 2
                            msg_bytes = pkt_body[pos:]
                            try:
                                data = json.loads(msg_bytes.decode("utf-8"))
                                if isinstance(data, dict):
                                    if "loona/command" in topic_str:
                                        cb = getattr(self, "on_loona_command", None)
                                        if callable(cb):
                                            cb(data)
                                    else:
                                        self.update_lightshow_state(data)
                            except Exception:
                                pass
            except Exception:
                self.lightshow_state["mqtt_connected"] = False
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass
            time.sleep(5.0)

    # --- Project-Specific Composite Pipelines ---

    def pipeline_ambiled(
        self,
        img: Image.Image,
        include_segmentation: bool = False,
        model_name: str = "ssd_coco",
        priority: str = "background",
    ) -> Dict[str, Any]:
        """AmbiLED (rpmlight) TPU pipeline:
        - Defaults to model_name='ssd_coco' & priority='background' for 100% SRAM cache sharing with Loona/Drones.
        - Computes TV screen bounds, active_content_bbox_norm (black-bar luma detector), scene_class, and wing_weights.
        """
        t0 = time.perf_counter()
        rgb = img.convert("RGB")
        det = self.detect(rgb, model_name=model_name, threshold=0.25, client="ambiled", priority=priority)
        tv_boxes = [d for d in det["detections"] if d["label"] in ("tv", "laptop")]
        people = [d for d in det["detections"] if d["label"] in ("person", "face")]
        action_objs = [
            d
            for d in det["detections"]
            if d["label"] in ("sports ball", "car", "motorcycle", "airplane", "boat", "skateboard", "surfboard")
        ]

        seg_regions = []
        if include_segmentation:
            seg = self.segment(rgb, client="ambiled", priority="normal")
            seg_regions = seg["regions"]
            if not tv_boxes:
                tv_boxes = [r for r in seg_regions if r["label"] == "tv"]

        # Fast NumPy luma thresholding on image to measure active non-black content bbox (2.35:1 / 4:3 / Kodi bars)
        arr = np.asarray(rgb, dtype=np.uint8)
        luma = (arr[:, :, 0].astype(np.uint16) * 77 + arr[:, :, 1].astype(np.uint16) * 150 + arr[:, :, 2].astype(np.uint16) * 29) >> 8
        row_max = np.percentile(luma, 92, axis=1)
        col_max = np.percentile(luma, 92, axis=0)
        active_rows = np.nonzero(row_max > 22)[0]
        active_cols = np.nonzero(col_max > 22)[0]
        h_px, w_px = luma.shape
        if active_rows.size > 0 and active_cols.size > 0:
            active_bbox = [
                round(float(active_cols[0]) / w_px, 4),
                round(float(active_rows[0]) / h_px, 4),
                round(float(active_cols[-1] + 1) / w_px, 4),
                round(float(active_rows[-1] + 1) / h_px, 4),
            ]
        else:
            active_bbox = [0.0, 0.0, 1.0, 1.0]

        # Compute focal saliency center weighted by detection score * sqrt(area)
        weighted_x, weighted_y, total_w = 0.5, 0.5, 0.0
        max_person_area = 0.0
        for d in det["detections"]:
            x1, y1, x2, y2 = d["bbox_norm"]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            area = max(0.01, (x2 - x1) * (y2 - y1))
            if d["label"] in ("person", "face") and area > max_person_area:
                max_person_area = area
            w = d["score"] * math.sqrt(area)
            weighted_x += cx * w
            weighted_y += cy * w
            total_w += w
        if total_w > 0:
            focal_cx = round((weighted_x - 0.5) / total_w, 4)
            focal_cy = round((weighted_y - 0.5) / total_w, 4)
        else:
            focal_cx, focal_cy = 0.5, 0.5

        if 1 <= len(people) <= 2 and max_person_area >= 0.18 and not action_objs:
            scene_class = "drama"
        elif len(people) >= 4 or len(action_objs) >= 1:
            scene_class = "music_video"
        else:
            scene_class = "auto"

        # Derive West / East / Far-Left Kitchen wing weights (0.5..1.5) from horizontal focal center
        bias = (focal_cx - 0.5) * 0.8  # -0.4 (left) .. +0.4 (right)
        wing_weights = {
            "west": round(float(np.clip(1.0 - bias, 0.55, 1.45)), 3),
            "east": round(float(np.clip(1.0 + bias, 0.55, 1.45)), 3),
            "far_left": round(float(np.clip(1.0 - bias * 1.25, 0.50, 1.50)), 3),
        }

        scene_cue = {
            "scene_class": scene_class,
            "active_content_bbox_norm": active_bbox,
            "wing_weights": wing_weights,
            "focal_x": focal_cx,
            "focal_y": focal_cy,
        }
        self.lightshow_state["scene_class"] = scene_class

        res = {
            "project": "AmbiLED (RPMLight)",
            "model": det["model"],
            "priority": priority,
            "sram_cache_hit": det["sram_cache_hit"],
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "tpu_ms": det["tpu_ms"],
            "tv_screen_detected": len(tv_boxes) > 0,
            "recommended_screen_bbox_norm": tv_boxes[0]["bbox_norm"] if tv_boxes else active_bbox,
            "viewers_in_room": len(people),
            "saliency_center_norm": {"x": focal_cx, "y": focal_cy},
            "scene_cue": scene_cue,
            "detections": det["detections"],
            "segmentation_regions": seg_regions,
        }
        self.stream_mailboxes["ambiled_cue"] = {**res, "updated_at": time.time()}
        return res

    def pipeline_loona(
        self,
        img: Image.Image,
        mode: str = "full",
        stream_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Loona pet robot pipeline.
        - mode='full' (Cue Mode, ~54ms): Runs ssd_face + movenet_pose (17 keypoints/gesture) + mobiledet_coco + optional Re-ID.
        - mode='fast' (Driving Stream Mode, ~5.9ms): Runs single SRAM-resident ssd_coco pass for 165 FPS wheel/head servoing.
        - Includes lightshow_dance_cue with automatic obstacle safety interlock.
        """
        t0 = time.perf_counter()
        if mode == "fast":
            objs = self.detect(
                img,
                model_name="ssd_coco",
                threshold=0.25,
                client="loona",
                priority="realtime",
                stream_id=stream_id or "loona_drive",
            )
            target_type = "none"
            target_cx, target_cy = 0.5, 0.5
            max_box_area = 0.0
            if objs["detections"]:
                for d in objs["detections"]:
                    x1, y1, x2, y2 = d["bbox_norm"]
                    max_box_area = max(max_box_area, (x2 - x1) * (y2 - y1))
                preferred = [
                    d for d in objs["detections"] if d["label"] in ("person", "cat", "dog", "sports ball")
                ]
                top_obj = preferred[0] if preferred else objs["detections"][0]
                x1, y1, x2, y2 = top_obj["bbox_norm"]
                target_cx, target_cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                target_type = f"fast:{top_obj['label']}"

            safe_wheels = max_box_area < 0.35
            res = {
                "project": "Loona Pet Robot",
                "mode": "fast",
                "stream_id": stream_id or "loona_drive",
                "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "tpu_ms": objs["tpu_ms"],
                "sram_cache_hit": objs["sram_cache_hit"],
                "gesture": "fast_mode_skipped",
                "recognized_person": None,
                "interaction_servo": {
                    "target_type": target_type,
                    "target_x_norm": round(target_cx, 4),
                    "target_y_norm": round(target_cy, 4),
                    "pan_error": round((target_cx - 0.5) * 2.0, 4),
                    "tilt_error": round((target_cy - 0.5) * 2.0, 4),
                },
                "lightshow_dance_cue": self.get_lightshow_cue(safe_for_wheel_motion=safe_wheels),
                "faces": [],
                "pose_keypoints": [],
                "objects": objs["detections"],
            }
            if stream_id:
                self.stream_mailboxes[stream_id] = {**res, "updated_at": time.time()}
            return res

        faces = self.detect(
            img, model_name="ssd_face", threshold=0.3, client="loona", priority="realtime", stream_id=stream_id
        )
        pose = self.estimate_pose(img, threshold=0.2, client="loona", priority="realtime")
        objs = self.detect(
            img, model_name="mobiledet_coco", threshold=0.3, client="loona", priority="realtime", stream_id=stream_id
        )

        # Determine primary social interaction target for Loona's head/ears/wheels
        target_type = "none"
        target_cx, target_cy = 0.5, 0.5
        recognized_person = None
        max_box_area = 0.0
        for d in objs["detections"]:
            x1, y1, x2, y2 = d["bbox_norm"]
            max_box_area = max(max_box_area, (x2 - x1) * (y2 - y1))

        if faces["detections"]:
            top_face = faces["detections"][0]
            x1, y1, x2, y2 = top_face["bbox_norm"]
            target_cx, target_cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            target_type = "face"
            if self.memory_bank:
                bx1, by1, bx2, by2 = [int(v) for v in top_face["bbox"]]
                crop = img.crop((bx1, by1, max(bx1 + 8, bx2), max(by1 + 8, by2)))
                match_res = self.match_memory(crop, category="person", client="loona")
                recognized_person = match_res["best_match"]
        elif pose["keypoints_by_name"]["nose"]["visible"]:
            nose = pose["keypoints_by_name"]["nose"]
            target_cx, target_cy = nose["x_norm"], nose["y_norm"]
            target_type = "person_pose"
        elif objs["detections"]:
            top_obj = objs["detections"][0]
            x1, y1, x2, y2 = top_obj["bbox_norm"]
            target_cx, target_cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            target_type = f"object:{top_obj['label']}"

        safe_wheels = max_box_area < 0.35
        res = {
            "project": "Loona Pet Robot",
            "mode": "full",
            "stream_id": stream_id,
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "tpu_ms": round(faces["tpu_ms"] + pose["tpu_ms"] + objs["tpu_ms"], 2),
            "gesture": pose["gesture"],
            "recognized_person": recognized_person,
            "interaction_servo": {
                "target_type": target_type,
                "target_x_norm": round(target_cx, 4),
                "target_y_norm": round(target_cy, 4),
                "pan_error": round((target_cx - 0.5) * 2.0, 4),
                "tilt_error": round((target_cy - 0.5) * 2.0, 4),
            },
            "lightshow_dance_cue": self.get_lightshow_cue(safe_for_wheel_motion=safe_wheels),
            "faces": faces["detections"],
            "pose_keypoints": pose["keypoints"],
            "objects": objs["detections"],
        }
        if stream_id:
            self.stream_mailboxes[stream_id] = {**res, "updated_at": time.time()}
        return res

    def pipeline_drones(
        self,
        img: Image.Image,
        target_class: str = "person",
        model_name: str = "ssd_coco",
        stream_id: Optional[str] = None,
        camera_mount: str = "forward",
    ) -> Dict[str, Any]:
        """Drones real-time visual servoing & obstacle telemetry pipeline."""
        t0 = time.perf_counter()
        det = self.detect(
            img,
            model_name=model_name,
            threshold=0.25,
            client="drones",
            priority="realtime",
            stream_id=stream_id,
        )
        targets = [d for d in det["detections"] if d["label"] == target_class]
        if not targets and det["detections"]:
            targets = det["detections"]

        servo = {
            "target_locked": False,
            "target_label": None,
            "yaw_error": 0.0,
            "roll_error": 0.0,
            "pitch_error": 0.0,
            "bbox_area_ratio": 0.0,
            "recommended_action": "hover_search",
        }
        if targets:
            primary = targets[0]
            x1, y1, x2, y2 = primary["bbox_norm"]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            area = (x2 - x1) * (y2 - y1)
            yaw_err = (cx - 0.5) * 2.0
            pitch_err = (cy - 0.5) * 2.0
            if area > 0.38:
                action = "back_off_obstacle_close"
            elif abs(yaw_err) > 0.18:
                action = "yaw_right" if yaw_err > 0 else "yaw_left"
            elif area < 0.12:
                action = "advance_track"
            else:
                action = "hold_station"
            servo = {
                "target_locked": True,
                "target_label": primary["label"],
                "score": primary["score"],
                "yaw_error": round(yaw_err, 4),
                "roll_error": round(yaw_err, 4),
                "pitch_error": round(pitch_err, 4),
                "bbox_area_ratio": round(area, 4),
                "recommended_action": action,
            }

        drone_id = 2 if stream_id and "2" in stream_id else (3 if stream_id and "3" in stream_id else 1)
        safe_to_dance = float(servo.get("bbox_area_ratio", 0.0)) < 0.35
        res = {
            "project": "Drones Visual Servoing",
            "stream_id": stream_id,
            "camera_mount": camera_mount,
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "tpu_ms": det["tpu_ms"],
            "sram_cache_hit": det["sram_cache_hit"],
            "flight_control": servo,
            "flight_servo": servo,
            "choreography": self._compute_swarm_choreography(drone_id=drone_id, safe_to_dance=safe_to_dance),
            "detections": det["detections"],
        }
        if stream_id:
            self.stream_mailboxes[stream_id] = {**res, "updated_at": time.time()}
        return res

    def pipeline_drones_optical_flow(
        self,
        img: Image.Image,
        stream_id: str = "drone1_down",
    ) -> Dict[str, Any]:
        """0% Coral TPU Downward Belly-Camera Optical Flow & High-Contrast Landing Pad Tracker on i3-N300 CPU."""
        t0 = time.perf_counter()
        gray = np.asarray(img.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
        now = time.perf_counter()

        # 1. 2D FFT Phase Correlation Optical Flow against previous frame for this stream_id
        flow_dx, flow_dy, confidence = 0.0, 0.0, 0.0
        prev_entry = self._optflow_prev.get(stream_id)
        if prev_entry is not None:
            prev_gray, prev_ts = prev_entry
            dt = max(1e-3, now - prev_ts)
            if dt < 2.0:
                # Hann window to suppress edge artifacts
                win = np.outer(np.hanning(64), np.hanning(64)).astype(np.float32)
                f1 = np.fft.rfft2((prev_gray - np.mean(prev_gray)) * win)
                f2 = np.fft.rfft2((gray - np.mean(gray)) * win)
                cross = f1 * np.conj(f2)
                denom = np.abs(cross) + 1e-8
                corr = np.fft.irfft2(cross / denom, s=(64, 64))
                peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
                confidence = float(corr[peak_y, peak_x])
                dy = float(peak_y - 64 if peak_y > 32 else peak_y)
                dx = float(peak_x - 64 if peak_x > 32 else peak_x)
                flow_dx = round(dx / 32.0, 4)
                flow_dy = round(dy / 32.0, 4)

        self._optflow_prev[stream_id] = (gray, now)

        # 2. High-contrast landing pad / bullseye centroid detector
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        pad_locked = False
        roll_err, pitch_err = -flow_dx, -flow_dy
        if std_val > 18.0:
            bright_mask = gray > (mean_val + 1.6 * std_val)
            frac = float(np.mean(bright_mask))
            if 0.015 <= frac <= 0.35:
                ys, xs = np.nonzero(bright_mask)
                cx_norm = float(np.mean(xs)) / 64.0
                cy_norm = float(np.mean(ys)) / 64.0
                roll_err = round((cx_norm - 0.5) * 2.0, 4)
                pitch_err = round((cy_norm - 0.5) * 2.0, 4)
                pad_locked = True

        servo = {
            "target_locked": pad_locked or (confidence > 0.15),
            "target_label": "landing_pad" if pad_locked else "optical_flow_hold",
            "score": round(confidence, 4),
            "roll_error": round(roll_err, 4),
            "pitch_error": round(pitch_err, 4),
            "yaw_error": 0.0,
            "flow_vx_norm": flow_dx,
            "flow_vy_norm": flow_dy,
            "recommended_action": "align_landing_pad" if pad_locked else "damped_position_hold",
        }
        drone_id = 2 if stream_id and "2" in stream_id else (3 if stream_id and "3" in stream_id else 1)
        cpu_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        res = {
            "project": "Drones Downward Optical Flow & Pad Tracker (CPU/iGPU)",
            "stream_id": stream_id,
            "camera_mount": "down",
            "tpu_ms": 0.0,
            "cpu_ms": cpu_ms,
            "total_ms": cpu_ms,
            "sram_cache_hit": True,
            "flight_control": servo,
            "flight_servo": servo,
            "choreography": self._compute_swarm_choreography(drone_id=drone_id, safe_to_dance=True),
        }
        self.stream_mailboxes[stream_id] = {**res, "updated_at": time.time()}
        return res

    def pipeline_cameras(
        self,
        img: Image.Image,
        camera_id: str = "front_porch",
        threshold: float = 0.30,
    ) -> Dict[str, Any]:
        """Smart NVR / Home Assistant Camera Detection pipeline (YOLOv9s + Face Crop + Re-ID)."""
        t0 = time.perf_counter()
        det = self.detect(img, model_name="yolov9s", threshold=threshold, client="cameras", priority="high")
        faces = []
        people = [d for d in det["detections"] if d["label"] == "person"]
        if people:
            face_res = self.detect(img, model_name="ssd_face", threshold=0.35, client="cameras", priority="high")
            faces = face_res["detections"]

        counts_by_label: Dict[str, int] = {}
        for d in det["detections"]:
            lbl = d["label"]
            counts_by_label[lbl] = counts_by_label.get(lbl, 0) + 1

        return {
            "project": "Camera Detections",
            "camera_id": camera_id,
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "tpu_ms": det["tpu_ms"],
            "sram_cache_hit": det["sram_cache_hit"],
            "occupied": "person" in counts_by_label,
            "counts": counts_by_label,
            "detections": det["detections"],
            "faces": faces,
        }

    def render_annotated_jpeg(
        self,
        img: Image.Image,
        detections: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> bytes:
        canvas = img.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        palette = {
            "person": "#10b981",
            "face": "#f59e0b",
            "vehicle": "#3b82f6",
            "car": "#3b82f6",
            "animal": "#ec4899",
            "dog": "#ec4899",
            "cat": "#ec4899",
            "tv": "#8b5cf6",
        }
        if detections:
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                color = palette.get(d["label"], "#22d3ee")
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                caption = f"{d['label']} {int(d['score'] * 100)}%"
                draw.rectangle([x1, max(0, y1 - 18), x1 + len(caption) * 7 + 8, y1], fill=color)
                draw.text((x1 + 4, max(0, y1 - 16)), caption, fill="#000000")

        if keypoints:
            edges = [
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            for i, j in edges:
                if i < len(keypoints) and j < len(keypoints):
                    kp1, kp2 = keypoints[i], keypoints[j]
                    if kp1.get("visible") and kp2.get("visible"):
                        draw.line([kp1["x"], kp1["y"], kp2["x"], kp2["y"]], fill="#f43f5e", width=3)
            for kp in keypoints:
                if kp.get("visible"):
                    r = 4
                    draw.ellipse([kp["x"] - r, kp["y"] - r, kp["x"] + r, kp["y"] + r], fill="#fde047", outline="#000")

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=88)
        return buf.getvalue()
