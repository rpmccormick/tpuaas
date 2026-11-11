# TPUaaS — Agent Integration Guide (`v1.2.0`)

> **Target Audience**: AI Coding Agents & Autonomous Control Services working on **AmbiLED (`rpmlight`)**, **Autonomous Pet Robots (Loona)**, **Autonomous Drone Fleets**, **Camera Detections (Home Assistant / NVR)**, or new vision projects.
> **Feature Requests & Coordination**: Post your requests in [`AGENT_BOARD.md`](AGENT_BOARD.md) or via `POST /v1/board`.

---

## 1. Service Endpoints & Discovery

| Property | Endpoint / Value |
| :--- | :--- |
| **Base URL** | `http://<tpu-host>:8090` (configured via `TPUAAS_URL`) |
| **WebSocket URL** | `ws://<tpu-host>:8090/ws/stream` |
| **Live Web Dashboard** | `GET /` |
| **OpenAPI / Swagger Schema** | `GET /docs` |
| **Machine-Readable Agent Guide** | `GET /AGENT_GUIDE.md` or `GET /llms.txt` |
| **Multi-Agent Coordination Board** | `GET /AGENT_BOARD.md` (`GET/POST /v1/board`) |
| **Hardware Accelerator** | Google Coral Edge TPU (PCIe `/dev/apex_0` or USB, 8 MB SRAM) + Host CPU Optical Flow |
| **Systemd Service** | `tpuaas.service` (`/opt/tpuaas/server.py`, configured via `/opt/tpuaas/.env`) |

---

## 2. Two Operating Modes: Cue Mode vs. Direct Stream Mode

TPUaaS is architected around **two distinct operating modes** so slow snapshot/cue tasks and high-frequency robot/drone control loops can share one physical Coral Edge TPU without blocking or causing control-loop bufferbloat.

### Mode 1: Queued Image-to-Output ("Slow / Cue Mode")
- **When to use**: Camera NVR alerts, Home Assistant automations, AmbiLED random-frame room/TV scene cues, and rich multi-model social cues (Face + 17-Keypoint Pose + COCO Objects + Face Re-ID).
- **How it works**:
  - Omit `stream_id` (default for all `POST /v1/...` calls).
  - Requests enter the **Priority + SRAM-Locality Queue** (`priority=realtime|high|normal|background`).
  - **Anti-Starvation Priority Aging (`85 ms`)**: Any Mode 1 task waiting `>= 85 ms` automatically promotes to `priority = -1` (above `realtime = 0`) so slow background tasks never starve when high-rate streams are active.

### Mode 2: Direct Real-Time Stream ("Fast Control-Loop Mode")
- **When to use**: Driving a robot or flying drones where stale frames cause PID oscillation.
- **How it works**:
  1. **1-Slot Latest-Frame Overwrite Mailbox (`stream_id`)**:
     - Pass a unique `stream_id` (e.g. `stream_id=loona_drive`, `drone1_fwd`, `drone2_fwd`, `drone3_fwd`) via `WS /ws/stream` or `POST /v1/projects/...`.
     - If a camera pushes frames faster than the TPU cycle, any unstarted frame for that `stream_id` is **overwritten in-place** by the newest frame (`dropped_stale_stream_frames`).
  2. **Non-Blocking Mailbox Polling (`GET /v1/streams/{stream_id}`)**:
     - Your motor/flight PID loop can read the latest inference result at any frequency via `GET /v1/streams/{stream_id}` in `<0.5 ms` without waiting on an inference round-trip.
  3. **SRAM Model-Pinning (`mode=fast` / `model=ssd_coco`)**:
     - Keeps `ssd_coco` (`5.94 ms` TPU time = **168 FPS hardware capacity**) resident in the EdgeTPU's 8 MB SRAM with **0.0 ms weight-swap overhead**.

---

## 3. Hardware Capacity Math: Robot + 3× Dual-Camera Drones + Ambilight on 1 TPU

- The Coral Edge TPU has **8 MB of on-chip SRAM**.
  - Consecutive requests on the **same model** (`sram_cache_hit = true`) take **5.2 ms – 10.0 ms**.
  - Switching between models that exceed 8 MB combined adds **~8 ms – 12 ms** of PCIe DMA weight-reload overhead.
- **Simultaneous Multi-Project Budget (`ssd_coco` pinned + CPU Optical Flow)**:
  - **3× Drone Forward Cams** (`drone1..3_fwd`, `ssd_coco`) @ **15 FPS each** = `45 FPS` (`267.3 ms/s` TPU time = **26.7%**)
  - **3× Drone Downward Belly Cams** (`drone1..3_down`, `/v1/projects/drones/optical_flow`) @ **15 FPS each** = **`0.0 ms/s` TPU time (0.0% TPU, runs in ~0.8 ms on host CPU)**
  - **1× Robot Fast Driving Stream** (`loona_drive`, `mode=fast`, `ssd_coco`) @ **18 FPS** = `18 FPS` (`106.9 ms/s` TPU time = **10.7%**)
  - **1× AmbiLED Random-Frame Scene Cue** (`ambiled`, `format=i420`, `ssd_coco`) @ **2.5 Hz** = `2.5 FPS` (`14.9 ms/s` TPU time = **1.5%**)
  - **Total Simultaneous Load**: **38.9% of 1 Coral Edge TPU (`61.1%` free headroom!)** with **100% SRAM cache hit rate**.

---

## 4. Pre-Loaded EdgeTPU Models (`GET /v1/models`)

| Model ID | Task | Input Tensor | Cached TPU Time | Classes / Output |
| :--- | :--- | :--- | :--- | :--- |
| `ssd_coco` | Fast COCO Detection | `300×300 uint8` | **5.94 ms** | 90 COCO classes (`person`, `tv`, `dog`, `cat`, `car`, `chair`, `sports ball`, etc.) |
| `mobiledet_coco` | High-mAP COCO Detection | `320×320 uint8` | **7.20 ms** | 90 COCO classes |
| `yolov9s` | Security Detection | `320×320 int8` | **10.01 ms** | 3 classes: `person`, `vehicle`, `animal` |
| `efficientdet_coco` | Precision COCO Detection | `320×320 uint8` | **11.50 ms** | 90 COCO classes |
| `ssd_face` | Human Face Detection | `320×320 uint8` | **5.20 ms** | Human face bounding boxes |
| `movenet_pose` | Human Pose & Gesture | `192×192 uint8` | **5.63 ms** | 17 keypoints + gesture (`hands_up`, `waving_left`, `waving_right`, `crouching_or_seated`, `standing_attentive`) |
| `deeplab_seg` | Semantic Segmentation | `513×513 uint8` | **44.61 ms** | 21 PASCAL VOC classes (`tv`, `person`, `sofa`, `chair`, `dog`, `cat`, etc.) |
| `mobilenet_cls` | ImageNet Classification | `224×224 uint8` | **2.75 ms** | 1000 ImageNet classes |
| `embeddings` | Feature Vector & Re-ID | `224×224 uint8` | **2.17 ms** | 1024-D L2-normalized vector + persistent cosine-similarity Memory Bank |

---

## 5. Endpoint Quick Reference (`v1.2.0`)

Every `POST` endpoint accepts images in **5 ways**:
1. **Raw planar `I420` (`320×176`, `84,480` bytes)** (`?format=i420&width=320&height=176` or auto-detected `84,480`-byte body) — *0.0 ms CPU overhead on embedded wall switches; vectorized NumPy decode in `<0.25 ms`*
2. **Raw binary JPEG/PNG/PPM bytes** (`Content-Type: image/jpeg` or `application/octet-stream`)
3. **Multipart form upload** (`file=@snapshot.jpg`)
4. **JSON body** (`{"image_url": "http://...", "image_base64": "..."}`)
5. **Query parameter** (`?image_url=http://...` or `?sample=pose|face|parrot`)

### A. Project-Specific Composite Endpoints
- **Autonomous Pet Robot (`loona`)**:
  - Cue Mode (Face + 17-KP Pose + Objects + Re-ID + Lightshow Dance Cue): `POST /v1/projects/loona?mode=full`
  - Fast Driving + Dance Stream Mode (~5.9ms `ssd_coco`): `POST /v1/projects/loona?mode=fast&stream_id=loona_drive`
- **Drones (3× Forward + 3× Downward Cameras)**:
  - Forward Camera Servoing + Swarm Dance Choreography (~5.9ms `ssd_coco`): `POST /v1/projects/drones?model=ssd_coco&target_class=person&stream_id=drone1_fwd&camera_mount=forward`
  - Downward Belly Camera Optical Flow + Landing Pad Bullseye (**0.0% Coral TPU, ~0.8ms CPU**): `POST /v1/projects/drones/optical_flow?stream_id=drone1_down`
- **AmbiLED (`rpmlight`)**:
  - Random-Frame TV Box + Active Non-Black Content BBox + `scene_class` (`drama`/`music_video`/`auto`) + `wing_weights` (`west`/`east`/`far_left`): `POST /v1/projects/ambiled?model=ssd_coco&priority=background&format=i420`
  - Non-blocking latest cue mailbox: `GET /v1/streams/ambiled_cue`
- **Cross-Project Music Video & Lightshow Bridge (`0.0%` Coral TPU)**:
  - Optional background MQTT listener (configured via `TPUAAS_MQTT_HOST` / `TPUAAS_MQTT_TOPICS` in `.env`).
  - Read current room palette, robot `dance_action`, and 3-drone `drone_swarm` choreography: `GET /v1/projects/lightshow`
  - Push explicit beat/energy/palette updates: `POST /v1/projects/lightshow`
- **Camera Detections**:
  - YOLOv9s + Conditional Face Detection: `POST /v1/projects/cameras?camera_id=front_porch&threshold=0.30`

### B. Shared-by-Default + On-Demand Exclusive Take-Over Leases
- **Acquire Exclusive Lease**: `POST /v1/lease/acquire` with JSON `{"client": "drones", "ttl_s": 30.0, "idle_timeout_s": 10.0, "pinned_model": "ssd_coco", "allow_same_model_sharing": true}`
- **Inspect / Release Lease**: `GET /v1/lease` and `POST /v1/lease/release`

### C. Universal Task Endpoints
- `POST /v1/detect?model=yolov9s|ssd_coco|mobiledet_coco|ssd_face&threshold=0.25&client=my_agent&priority=normal&stream_id=...`
- `POST /v1/pose?threshold=0.20&client=loona`
- `POST /v1/segment?include_mask_b64=false&client=ambiled`
- `POST /v1/classify?top_k=5&client=default`
- `POST /v1/embed`
- `POST /v1/memory/enroll?name=User&category=person` & `POST /v1/memory/match?category=person`
- `POST /v1/annotate?model=yolov9s&include_pose=true` (returns annotated `image/jpeg`)
- `GET /v1/streams` & `GET /v1/streams/{stream_id}` (latest non-blocking frame result for Mode 2 streams)
