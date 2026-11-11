# ⚡ TPUaaS — Coral EdgeTPU-as-a-Service (`v1.2.0`)

**Share a single Google Coral Edge TPU (`/dev/apex_0` PCIe M.2 or USB) across your entire homelab, robotics fleet, ambient lighting engine, and camera NVR — with zero model-thrashing and sub-6ms hardware latency.**

Normally, a Google Coral Edge TPU can only be opened by **one process at a time**, forcing homelabbers to lock their TPU inside a single container (like Frigate or Scrypted) while leaving robotics, TV ambilighting, pose estimation, and custom AI agents out in the cold. Worse, naively multiplexing multiple models across the Coral TPU's **8 MB on-chip SRAM** triggers ~10 ms PCIe DMA weight-reload penalties on every frame.

**TPUaaS** solves both problems by turning a single Coral Edge TPU into a multi-tenant, SRAM-locality-aware HTTP & WebSocket microservice with a built-in real-time control dashboard, 9 pre-allocated vision models, zero-copy raw `I420` ingestion, CPU-offloaded optical flow, and an agent-native coordination board.

---

## ✨ Key Architectural Highlights

1. **8 MB SRAM-Locality + Priority Scheduler (`168 FPS` Capacity)**:
   - Keeps all 9 EdgeTPU interpreters pre-allocated in a single hardware worker thread and prioritizes pending tasks that match the **currently cached SRAM model** (`sram_cache_hit = true`, **`5.94 ms`/frame** on `ssd_coco`, **`0.0 ms` weight-swap overhead**).
2. **Dual Operating Modes (Slow Queued Cues + Real-Time Control Streams)**:
   - **Mode 1 — Queued Image-to-Output (`priority=background|normal|high`)**: Every frame is guaranteed to complete in order. Includes **Anti-Starvation Priority Aging (`85 ms`)** so low-rate background tasks (e.g., random-frame TV Ambilight scene cues or NVR snapshots) automatically promote above continuous real-time streams if they wait `>= 85 ms`.
   - **Mode 2 — 1-Slot Latest-Frame-Wins Stream Mailboxes (`stream_id=...`)**: Designed for closed-loop robotics and drone PID controllers. If a camera pushes frames faster than the TPU cycle, unstarted frames for that `stream_id` are overwritten in-place (`dropped_stale_stream_frames`)—eliminating control-loop bufferbloat.
3. **Shared-by-Default + Watchdog-Protected Exclusive Take-Over Leases (`/v1/lease/*`)**:
   - Defaults to shared multiplexing (`shared_default`), while allowing high-priority flight or robotics controllers to demand an **Exclusive Take-Over Lease** (`POST /v1/lease/acquire`) guarded by a mandatory **TTL cap (`<= 300s`)** and **`10s` Idle Heartbeat Watchdog** so a crashed script can never permanently lock the TPU.
4. **Hybrid Coral TPU + CPU Optical Flow Offload**:
   - Offloads downward-facing drone belly cameras (`POST /v1/projects/drones/optical_flow`) to 64×64 Hann-windowed **2D FFT Phase Correlation optical flow** + landing-pad centroid tracking on the host CPU (`~0.8 ms`/frame, **0.0% Coral TPU usage**).
5. **Zero-Copy Raw Planar `I420` Ingestion**:
   - Accepts raw planar `I420` (`YUV420p`) video frames directly (`?format=i420&width=320&height=176`) and decodes them in `<0.25 ms` via vectorized NumPy—allowing embedded Linux panels (like NXP i.MX6 wall switches running [`rpmlight`](https://github.com/rpmccormick/rpmlight) / [`brilliant-mqtt`](https://github.com/rpmccormick/brilliant-mqtt)) to stream frames with `0.0 ms` local encoding overhead.
6. **Agent-Native Self-Discovery (`/AGENT_GUIDE.md`, `/llms.txt`, `/v1/board`)**:
   - Ships with machine-readable integration docs and a live multi-agent coordination board so AI coding agents can inspect bandwidth budgets and coordinate feature requests autonomously.

---

## 🧠 Pre-Loaded EdgeTPU Models (`GET /v1/models`)

| Model ID | Task | Input Tensor | Cached TPU Time | Output / Classes |
| :--- | :--- | :--- | :--- | :--- |
| `ssd_coco` | Fast Real-Time COCO Detection | `300×300 uint8` | **5.94 ms** (~168 FPS) | 90 COCO classes (`person`, `tv`, `dog`, `cat`, `car`, `sports ball`, etc.) |
| `mobiledet_coco` | High-mAP COCO Detection | `320×320 uint8` | **7.20 ms** (~138 FPS) | 90 COCO classes |
| `yolov9s` | Security Object Detection | `320×320 int8` | **10.01 ms** (~100 FPS) | 3 classes (`person`, `vehicle`, `animal`) with vectorized DFL+NMS decoder |
| `efficientdet_coco` | Precision COCO Detection | `320×320 uint8` | **11.50 ms** (~87 FPS) | 90 COCO classes |
| `ssd_face` | Human Face Detection | `320×320 uint8` | **5.20 ms** (~192 FPS) | Human face bounding boxes |
| `movenet_pose` | Human Pose & Gesture | `192×192 uint8` | **5.63 ms** (~177 FPS) | 17 body keypoints + high-level gesture (`hands_up`, `waving_left`, `waving_right`, `crouching_or_seated`, `standing_attentive`) |
| `deeplab_seg` | Semantic Segmentation | `513×513 uint8` | **44.61 ms** (~22 FPS) | 21 PASCAL VOC pixel-mask classes (`tv`, `person`, `sofa`, `chair`, etc.) |
| `mobilenet_cls` | ImageNet Classification | `224×224 uint8` | **2.75 ms** (~363 FPS) | 1000 ImageNet classes |
| `embeddings` | Feature Vector & Re-ID | `224×224 uint8` | **2.17 ms** (~460 FPS) | 1024-D L2-normalized embedding + persistent cosine-similarity Memory Bank |

---

## 🚀 Quick Start

### Option A: Debian 12 (Bookworm) / Ubuntu 22.04+ / Proxmox LXC (Turnkey Script)
Google's legacy `libedgetpu1-std` apt repository is broken on modern Debian/Ubuntu releases and `pycoral` requires Python 3.9. Our installer handles `libedgetpu.so.1.0`, standalone Python 3.9 via `uv`, model downloads, and `systemd` setup in one command:

```bash
git clone https://github.com/rpmccormick/tpuaas.git
cd tpuaas
sudo bash scripts/install_lxc.sh
```

### Option B: Docker Compose
```bash
git clone https://github.com/rpmccormick/tpuaas.git
cd tpuaas
cp .env.example .env
docker compose up -d --build
```

Once running, open **`http://<your-tpu-host>:8090/`** for the live dark-mode telemetry dashboard and interactive model playground, or **`http://<your-tpu-host>:8090/docs`** for the OpenAPI specification.

---

## 🔌 Built-In Reference Project Pipelines

In addition to universal task endpoints (`/v1/detect`, `/v1/pose`, `/v1/segment`, `/v1/classify`, `/v1/embed`, `/v1/memory/*`, `/v1/annotate`), TPUaaS includes four composite pipelines demonstrating how multiple real-time and background systems share **one Coral Edge TPU at <40% total load**:

1. **AmbiLED / [`rpmlight`](https://github.com/rpmccormick/rpmlight) (`POST /v1/projects/ambiled`)**:
   - Accepts raw `320×176` `I420` frames or JPEGs (`priority=background`, `model=ssd_coco`).
   - Returns TV boundary boxes, 92nd-percentile NumPy luma `active_content_bbox_norm` (letterbox/pillarbox black-bar detector), `scene_class` (`"drama" | "music_video" | "auto"`), and horizontal saliency `wing_weights` (`{"west", "east", "far_left"}`).
2. **Autonomous Pet Robot (`POST /v1/projects/loona`)**:
   - `mode=fast` (`~5.9 ms` `ssd_coco` stream mode for 165 FPS wheel/head visual servoing) or `mode=full` (`~54 ms` multi-model Face + 17-Keypoint Pose + COCO Objects + Face Re-ID).
   - Includes `lightshow_dance_cue` with automatic obstacle safety interlock (`safe_for_wheel_motion`).
3. **Multi-Drone Visual Servoing + CPU Optical Flow (`POST /v1/projects/drones` & `/v1/projects/drones/optical_flow`)**:
   - Forward camera obstacle/target servoing on the Coral TPU (`~5.9 ms`) + downward belly camera 2D FFT Phase Correlation optical flow and landing-pad lock on the CPU (`0.0%` TPU load), plus 120°-phase-offset 3-drone swarm dance choreography.
4. **Smart Camera NVR (`POST /v1/projects/cameras`)**:
   - Runs `yolov9s` (`person`, `vehicle`, `animal`) and conditionally triggers `ssd_face` when people are present.

---

## 🐍 Zero-Dependency Python SDK (`client.py`)

```python
from client import TPUaaSClient

# Reads TPUAAS_URL from environment/.env (defaults to http://localhost:8090)
tpu = TPUaaSClient("http://localhost:8090", client_name="my_project")

# 1. Fast Real-Time Stream Mode (Mode 2: 1-slot overwrite mailbox, ~5.9ms)
servo = tpu.drones(jpeg_bytes, model="ssd_coco", stream_id="drone1_fwd")
print(servo["flight_servo"], servo["tpu_ms"])

# 2. 0% TPU Downward Belly Camera Optical Flow (~0.8ms CPU)
flow = tpu.drones_optical_flow(belly_jpeg_bytes, stream_id="drone1_down")
print(flow["flight_servo"]["flow_vx_norm"])

# 3. Zero-Copy Raw I420 Ambilight Scene Cue (Mode 1: Queued with 85ms anti-starvation aging)
cue = tpu.ambiled(raw_i420_bytes, fmt="i420", width=320, height=176)
print(cue["scene_cue"]["scene_class"], cue["scene_cue"]["wing_weights"])
```

---

## 📄 License

MIT License — Copyright (c) 2026 Ryan McCormick ([@rpmccormick](https://github.com/rpmccormick)).
