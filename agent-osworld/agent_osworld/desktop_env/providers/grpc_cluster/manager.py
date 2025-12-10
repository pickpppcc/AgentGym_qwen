import os
from typing import List

from desktop_env.providers.base import VMManager


class GrpcClusterVMManager(VMManager):
    """
    Minimal VM manager for grpc_cluster provider.
    It doesn't manage local VM files; it delegates lifecycle to a remote controller.
    """

    def __init__(self):
        self._registered: List[str] = []

    def initialize_registry(self, **kwargs):
        # No-op for remote cluster; keep an in-memory placeholder to satisfy interface
        pass

    def add_vm(self, vm_path, **kwargs):
        if vm_path and vm_path not in self._registered:
            self._registered.append(vm_path)

    def delete_vm(self, vm_path, **kwargs):
        if vm_path in self._registered:
            self._registered.remove(vm_path)

    def occupy_vm(self, vm_path, pid, **kwargs):
        # No-op in remote mode; controller handles resource allocation
        pass

    def list_free_vms(self, **kwargs):
        # Remote cluster capacity is dynamic; return placeholder list
        return self._registered

    def check_and_clean(self, **kwargs):
        # Nothing to clean locally
        pass

    def get_vm_path(self, **kwargs):
        # Return an image identifier if provided, otherwise a placeholder
        # DesktopEnv passes this to provider.start_emulator but grpc_cluster can ignore it
        return os.environ.get("OSWORLD_VM_IMAGE", "remote:default")


