import os
import logging
import time
from typing import Optional

import requests

from desktop_env.providers.base import Provider


logger = logging.getLogger("desktopenv.providers.grpc.GrpcClusterProvider")
logger.setLevel(logging.INFO)


class GrpcClusterProvider(Provider):
    """
    Provider that delegates VM lifecycle to a remote Controller via HTTP/gRPC gateway.
    For MVP we assume a RESTful gateway exists at OSWORLD_CONTROLLER_URL to reduce deps.
    Later this can be replaced by native gRPC client calls.
    """

    def __init__(self, region: str):
        super().__init__(region)
        self.controller_url: str = os.environ.get("OSWORLD_CONTROLLER_URL", "http://localhost:8088")
        self.env_id: Optional[str] = None
        self.server_port: Optional[int] = None
        self.chromium_port: Optional[int] = None
        self.vnc_port: Optional[int] = None
        self.vlc_port: Optional[int] = None
        self.vm_host: Optional[str] = None

    def _wait_ready(self, timeout: int = 300) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{self.controller_url}/env/{self.env_id}/health", timeout=(5, 5))
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError("Remote environment failed to become ready in time")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        # path_to_vm can be treated as image identifier in remote cluster
        payload = {
            "image": path_to_vm,
            "headless": headless,
            "os_type": os_type,
            "region": self.region,
        }
        logger.info(f"Requesting remote environment from controller {self.controller_url}...")
        r = requests.post(f"{self.controller_url}/env", json=payload, timeout=(10, 30))
        r.raise_for_status()
        data = r.json()
        self.env_id = data["env_id"]
        # Controller returns connection endpoints
        self.vm_host = data.get("host", "localhost")
        self.server_port = int(data.get("server_port", 5000))
        self.chromium_port = int(data.get("chromium_port", 9222))
        self.vnc_port = int(data.get("vnc_port", 8006))
        self.vlc_port = int(data.get("vlc_port", 8080))

        self._wait_ready()

    def get_ip_address(self, path_to_vm: str) -> str:
        if not self.env_id:
            raise RuntimeError("Remote environment not started")
        host = self.vm_host or "localhost"
        return f"{host}:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        if not self.env_id:
            return
        try:
            requests.post(f"{self.controller_url}/env/{self.env_id}/snapshot", json={"name": snapshot_name}, timeout=(5, 10))
        except Exception:
            pass

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        if not self.env_id:
            return path_to_vm
        try:
            requests.post(f"{self.controller_url}/env/{self.env_id}/revert", json={"name": snapshot_name}, timeout=(5, 30))
        except Exception:
            pass
        # path_to_vm remains an identifier on controller side
        return path_to_vm

    def stop_emulator(self, path_to_vm: str):
        if not self.env_id:
            return
        try:
            requests.delete(f"{self.controller_url}/env/{self.env_id}", timeout=(5, 10))
        except Exception:
            pass
        finally:
            self.env_id = None
            self.server_port = None
            self.chromium_port = None
            self.vnc_port = None
            self.vlc_port = None
            self.vm_host = None


