import os
from functools import wraps

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_GDN_FIX_BT = os.getenv("FLA_GDN_FIX_BT", "0") == "1"

SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))

use_cuda_graph = os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"

is_gather_supported = hasattr(triton.language, "gather")

is_nvidia_hopper = torch.cuda.get_device_capability()[0] >= 9

is_tma_supported = (is_nvidia_hopper) and (
    hasattr(triton.language, "_experimental_make_tensor_descriptor")
    or hasattr(triton.language, "make_tensor_descriptor")
)


if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
else:
    exp = tl.exp


# Simple helper to check whether the current CUDA device(s) have enough
# shared memory for larger BS choices. This mirrors the original project's
# intent (different BS_LIST when devices have more shared memory).
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        # torch.cuda.get_device_properties exists when CUDA is available
        import torch

        if not torch.cuda.is_available():
            return False
        prop = torch.cuda.get_device_properties(tensor_idx)
        # property names differ across torch versions/drivers; try common ones
        max_shared = getattr(prop, "max_shared_memory_per_multiprocessor", None)
        if max_shared is None:
            max_shared = getattr(prop, "max_shared_memory", None)
        if max_shared is None:
            # fallback conservative default
            return False
        # Use the AMPERE threshold used in the original project as heuristic
        return max_shared >= 166_000
    except Exception:
        return False


# Lightweight input guard decorator used across kernels to surface
# clearer error messages or perform light checks. The original project
# provides a more featureful guard; here we keep a minimal, safe passthrough
# implementation so migrated modules can import and use `@input_guard`.


def input_guard(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Minimal checks can be added here if needed (e.g., device checks)
        return fn(*args, **kwargs)

    return wrapper
