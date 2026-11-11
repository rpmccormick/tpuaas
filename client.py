"""Zero-dependency Python SDK for TPUaaS (Coral EdgeTPU-as-a-Service)."""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Union


def _load_dotenv() -> None:
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


class TPUaaSClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        client_name: str = "default",
        timeout: float = 5.0,
    ):
        resolved_url = base_url or os.environ.get("TPUAAS_URL", "http://localhost:8090")
        self.base_url = resolved_url.rstrip("/")
        self.client_name = client_name
        self.timeout = timeout

    def _post_image(
        self,
        path: str,
        image: Union[bytes, str, Path, None] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        query = dict(params or {})
        query.setdefault("client", self.client_name)
        q_str = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        url = f"{self.base_url}{path}?{q_str}" if q_str else f"{self.base_url}{path}"

        if isinstance(image, Path):
            data = image.read_bytes()
            headers = {"Content-Type": "application/octet-stream"}
        elif isinstance(image, (bytes, bytearray)):
            data = bytes(image)
            headers = {"Content-Type": "application/octet-stream"}
        elif isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
            data = json.dumps({"image_url": image}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
        elif isinstance(image, str) and Path(image).exists():
            data = Path(image).read_bytes()
            headers = {"Content-Type": "application/octet-stream"}
        else:
            data = b""
            headers = {}

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def status(self) -> Dict[str, Any]:
        with urllib.request.urlopen(f"{self.base_url}/v1/status", timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def detect(
        self,
        image: Union[bytes, str, Path],
        model: str = "yolov9s",
        threshold: float = 0.25,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        return self._post_image(
            "/v1/detect",
            image=image,
            params={"model": model, "threshold": threshold, "priority": priority},
        )

    def pose(self, image: Union[bytes, str, Path], threshold: float = 0.20) -> Dict[str, Any]:
        return self._post_image("/v1/pose", image=image, params={"threshold": threshold})

    def segment(self, image: Union[bytes, str, Path], include_mask_b64: bool = False) -> Dict[str, Any]:
        return self._post_image("/v1/segment", image=image, params={"include_mask_b64": include_mask_b64})

    def ambiled(
        self,
        image: Union[bytes, str, Path],
        include_segmentation: bool = False,
        model: str = "ssd_coco",
        priority: str = "background",
        fmt: Optional[str] = None,
        width: int = 320,
        height: int = 176,
    ) -> Dict[str, Any]:
        return self._post_image(
            "/v1/projects/ambiled",
            image=image,
            params={
                "include_segmentation": include_segmentation,
                "model": model,
                "priority": priority,
                "format": fmt,
                "width": width,
                "height": height,
            },
        )

    def loona(
        self,
        image: Union[bytes, str, Path],
        mode: str = "full",
        stream_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._post_image(
            "/v1/projects/loona",
            image=image,
            params={"mode": mode, "stream_id": stream_id},
        )

    def drones(
        self,
        image: Union[bytes, str, Path],
        target_class: str = "person",
        model: str = "ssd_coco",
        stream_id: Optional[str] = None,
        camera_mount: str = "forward",
    ) -> Dict[str, Any]:
        return self._post_image(
            "/v1/projects/drones",
            image=image,
            params={
                "target_class": target_class,
                "model": model,
                "stream_id": stream_id,
                "camera_mount": camera_mount,
            },
        )

    def drones_optical_flow(
        self,
        image: Union[bytes, str, Path],
        stream_id: str = "drone1_down",
    ) -> Dict[str, Any]:
        return self._post_image(
            "/v1/projects/drones/optical_flow",
            image=image,
            params={"stream_id": stream_id},
        )

    def lightshow(self, safe_for_wheel_motion: bool = True) -> Dict[str, Any]:
        q_str = urllib.parse.urlencode({"safe_for_wheel_motion": safe_for_wheel_motion})
        with urllib.request.urlopen(f"{self.base_url}/v1/projects/lightshow?{q_str}", timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def acquire_lease(
        self,
        ttl_s: float = 30.0,
        idle_timeout_s: float = 10.0,
        pinned_model: str = "ssd_coco",
        allow_same_model_sharing: bool = True,
    ) -> Dict[str, Any]:
        payload = json.dumps(
            {
                "client": self.client_name,
                "ttl_s": ttl_s,
                "idle_timeout_s": idle_timeout_s,
                "pinned_model": pinned_model,
                "allow_same_model_sharing": allow_same_model_sharing,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/lease/acquire",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def release_lease(self) -> Dict[str, Any]:
        payload = json.dumps({"client": self.client_name}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/lease/release",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_board_request(
        self,
        project: str,
        title: str,
        details: str,
        mode: str = "Mode 1 (Queued Cue)",
        budget: str = "unspecified",
    ) -> Dict[str, Any]:
        payload = json.dumps(
            {
                "agent": self.client_name,
                "project": project,
                "title": title,
                "mode": mode,
                "budget": budget,
                "details": details,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/board",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def cameras(
        self,
        image: Union[bytes, str, Path],
        camera_id: str = "front_porch",
        threshold: float = 0.30,
    ) -> Dict[str, Any]:
        return self._post_image(
            "/v1/projects/cameras",
            image=image,
            params={"camera_id": camera_id, "threshold": threshold},
        )
