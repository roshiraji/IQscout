"""Windows DDP bootstrap: disable libuv TCPStore (PyTorch wheels lack libuv).

Ultralytics launches ``python -m torch.distributed.run ...``. On Windows CUDA
wheels, TCPStore defaults to use_libuv=True and ignores USE_LIBUV=0. This
entry point patches TCPStore before torch.distributed.run starts rendezvous.
"""

from __future__ import annotations

import sys


def _patch_tcpstore() -> None:
    import torch.distributed as dist

    _Orig = dist.TCPStore

    def _TCPStore(*args, **kwargs):  # noqa: N802 — match torch API name
        kwargs["use_libuv"] = False
        return _Orig(*args, **kwargs)

    dist.TCPStore = _TCPStore  # type: ignore[misc, assignment]

    # Modules that did ``from torch.distributed import TCPStore`` keep a stale
    # binding — re-point those after import torch.distributed.
    for name in (
        "torch.distributed.rendezvous",
        "torch.distributed.elastic.rendezvous.static_tcp_rendezvous",
    ):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "TCPStore"):
            mod.TCPStore = _TCPStore  # type: ignore[attr-defined]


def main() -> None:
    _patch_tcpstore()
    # Import only after patch so StaticTCPRendezvous binds the wrapper.
    from torch.distributed.run import main as torchrun_main

    torchrun_main()


if __name__ == "__main__":
    main()
