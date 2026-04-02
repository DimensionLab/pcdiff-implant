"""
Backend loader for PVCNN operations.

Automatically selects between CUDA JIT-compiled backend (for NVIDIA GPUs)
and pure PyTorch CPU fallback (for macOS, CPU-only systems).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_real_backend = None
_backend_type = None  # 'cuda' or 'cpu'


def _load_cuda_backend():
    """Load the JIT-compiled CUDA backend."""
    import shutil

    from torch.utils.cpp_extension import load

    _src_path = os.path.dirname(os.path.abspath(__file__))

    # Auto-detect GCC path (prefer gcc-11 for CUDA 12.x compatibility, fall back to system gcc)
    _gcc_path = None
    for gcc_name in ["gcc-12", "gcc-11", "gcc-10", "gcc"]:
        found = shutil.which(gcc_name)
        if found:
            _gcc_path = found
            break

    _extra_cuda_cflags = []
    if _gcc_path:
        _extra_cuda_cflags.append(f"--compiler-bindir={_gcc_path}")

    backend = load(
        name="_pvcnn_backend",
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=_extra_cuda_cflags,
        sources=[
            os.path.join(_src_path, "src", f)
            for f in [
                "ball_query/ball_query.cpp",
                "ball_query/ball_query.cu",
                "grouping/grouping.cpp",
                "grouping/grouping.cu",
                "interpolate/neighbor_interpolate.cpp",
                "interpolate/neighbor_interpolate.cu",
                "interpolate/trilinear_devox.cpp",
                "interpolate/trilinear_devox.cu",
                "sampling/sampling.cpp",
                "sampling/sampling.cu",
                "voxelization/vox.cpp",
                "voxelization/vox.cu",
                "bindings.cpp",
            ]
        ],
    )
    return backend


def _load_cpu_backend():
    """Load the pure PyTorch CPU fallback backend."""
    from modules.functional.cpu_backend import _cpu_backend

    return _cpu_backend


def get_backend():
    """
    Get the appropriate backend for the current system.

    Returns the CUDA backend if CUDA is available, otherwise returns
    the CPU fallback backend.
    """
    global _real_backend, _backend_type

    if _real_backend is not None:
        return _real_backend

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()

    # Also check if CUDA_HOME is set (needed for JIT compilation)
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if cuda_available and cuda_home:
        try:
            logger.info("Loading CUDA backend for PVCNN operations...")
            _real_backend = _load_cuda_backend()
            _backend_type = "cuda"
            logger.info("CUDA backend loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load CUDA backend: {e}")
            logger.info("Falling back to CPU backend...")
            _real_backend = _load_cpu_backend()
            _backend_type = "cpu"
            logger.info("CPU backend loaded successfully.")
    else:
        if not cuda_available:
            logger.info("CUDA not available. Using CPU backend for PVCNN operations.")
        elif not cuda_home:
            logger.info("CUDA_HOME not set. Using CPU backend for PVCNN operations.")
        _real_backend = _load_cpu_backend()
        _backend_type = "cpu"
        logger.info("CPU backend loaded successfully.")

    return _real_backend


def get_backend_type():
    """Get the type of backend being used ('cuda' or 'cpu')."""
    global _backend_type
    if _backend_type is None:
        get_backend()  # Initialize if not already
    return _backend_type


# For backward compatibility, expose _backend at module level
# This will be lazily loaded on first access
class _BackendProxy:
    """Proxy object that lazily loads the backend on first attribute access."""

    def __getattr__(self, name):
        backend = get_backend()
        return getattr(backend, name)


_backend = _BackendProxy()

__all__ = ["_backend", "get_backend", "get_backend_type"]
