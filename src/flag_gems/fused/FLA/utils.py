<<<<<<< HEAD
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

=======
>>>>>>> df000e19 (update fla kernels)
import contextlib
import functools
import os
from collections.abc import Callable
from typing import Any

import torch
import triton

<<<<<<< HEAD
from flag_gems import runtime
from flag_gems.utils.device_info import get_device_capability

=======
>>>>>>> df000e19 (update fla kernels)
# envrironments setting
SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))
FLA_GDN_FIX_BT = os.getenv("FLA_GDN_FIX_BT", "0") == "1"

use_cuda_graph = os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"


<<<<<<< HEAD
def _detect_nvidia_hopper() -> bool:
    """Return True if current device is NVIDIA and SM major version >= 9.

    We rely on `runtime.device.vendor_name` and `get_device_capability()` which
    already handle errors and fallbacks elsewhere.
    """
    vendor_name = getattr(runtime.device, "vendor_name", "").lower()
    if "nvidia" not in vendor_name:
        return False
    major, _ = get_device_capability()
    return major >= 9


is_nvidia_hopper = _detect_nvidia_hopper()

is_tma_supported = is_nvidia_hopper and (
=======
is_nvidia_hopper = torch.cuda.get_device_capability()[0] >= 9  # TODO

is_tma_supported = (is_nvidia_hopper) and (
>>>>>>> df000e19 (update fla kernels)
    hasattr(triton.language, "_experimental_make_tensor_descriptor")
    or hasattr(triton.language, "make_tensor_descriptor")
)


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries: tuple[tuple | None, dict | None, Any] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
<<<<<<< HEAD
        nonlocal cache_entries
=======
        nonlocal cache_entries, cache_size
>>>>>>> df000e19 (update fla kernels)
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:i]
                    + cache_entries[i + 1 :]
                    + [(args, kwargs, last_result)]
                )
                return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.cuda.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


<<<<<<< HEAD
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    from flag_gems.utils.device_info import get_device_properties

    props = get_device_properties()
    if props is None:
        return False

    # property names differ across torch versions/drivers; try common ones
    max_shared = getattr(props, "max_shared_memory_per_multiprocessor", None)
    if max_shared is None:
        max_shared = getattr(props, "max_shared_memory", None)
    if max_shared is None:
        # fallback conservative default
        return False
    # Use the AMPERE threshold used in the original project as heuristic
    return max_shared >= 166_000
=======
# TODO: fixme
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    return False  # TODO
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
>>>>>>> df000e19 (update fla kernels)
