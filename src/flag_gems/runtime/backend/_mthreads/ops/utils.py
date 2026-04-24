import os
from collections import OrderedDict

import numpy as np
import torch
import triton
import triton.language as tl

_TMA_DESCRIPTOR_CACHE_MAXSIZE = 256
_tma_descriptor_cache = OrderedDict()

# Detect once whether fill_2d_tma_descriptor expects a pointer (int) or numpy array.
# triton >= 3.2 changed the last parameter from numpy array to int pointer.
_fill_2d_tma = triton.runtime.driver.active.utils.fill_2d_tma_descriptor
_tma_desc_wants_ptr = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 2)


def _tma_desc_arg(desc_np):
    return int(desc_np.ctypes.data) if _tma_desc_wants_ptr else desc_np


def create_tma_device_descriptor_v31(tensor, block_m, block_n, device):
    assert tensor.dim() == 2, "TMA descriptor only supports 2D tensors"
    TMA_DESCRIPTOR_SIZE = 64
    desc_np = np.empty(TMA_DESCRIPTOR_SIZE, dtype=np.int8)
    shapes = [tensor.shape[0], tensor.shape[1]]
    if not tensor.is_contiguous():
        assert (
            tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
        ), "TMA descriptor only supports contiguous or transposed 2D tensors"
        shapes.reverse()
    _fill_2d_tma(
        tensor.data_ptr(),
        shapes[0],
        shapes[1],
        block_m,
        block_n,
        tensor.element_size(),
        _tma_desc_arg(desc_np),
    )
    desc = torch.tensor(desc_np, device=device)
    return desc


def create_tma_device_descriptor(tensor, block_m, block_n, device):
    assert tensor.dim() == 2, "TMA descriptor only supports 2D tensors"
    TMA_DESCRIPTOR_SIZE = 64
    desc_np = np.empty(TMA_DESCRIPTOR_SIZE, dtype=np.int8)
    shapes = [tensor.shape[0], tensor.shape[1]]
    if not tensor.is_contiguous():
        assert (
            tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
        ), "TMA descriptor only supports contiguous or transposed 2D tensors"
        shapes.reverse()
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        tensor.data_ptr(),
        shapes[0],
        shapes[1],
        block_m,
        block_n,
        tensor.element_size(),
        int(desc_np.ctypes.data),
    )
    desc = torch.tensor(desc_np, device=device)
    return desc


def _tma_descriptor_cache_key(tensor, block_m, block_n, device):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        block_m,
        block_n,
        str(device),
    )


def get_cached_tma_device_descriptor_v31(tensor, block_m, block_n, device):
    key = _tma_descriptor_cache_key(tensor, block_m, block_n, device)
    desc = _tma_descriptor_cache.get(key)
    if desc is not None:
        _tma_descriptor_cache.move_to_end(key)
        return desc

    desc = create_tma_device_descriptor_v31(tensor, block_m, block_n, device)
    _tma_descriptor_cache[key] = desc
    if len(_tma_descriptor_cache) > _TMA_DESCRIPTOR_CACHE_MAXSIZE:
        _tma_descriptor_cache.popitem(last=False)
    return desc


def get_cached_tma_device_descriptor(tensor, block_m, block_n, device):
    key = _tma_descriptor_cache_key(tensor, block_m, block_n, device)
    desc = _tma_descriptor_cache.get(key)
    if desc is not None:
        _tma_descriptor_cache.move_to_end(key)
        return desc

    desc = create_tma_device_descriptor(tensor, block_m, block_n, device)
    _tma_descriptor_cache[key] = desc
    if len(_tma_descriptor_cache) > _TMA_DESCRIPTOR_CACHE_MAXSIZE:
        _tma_descriptor_cache.popitem(last=False)
    return desc


def get_triton_dtype(dtype):
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    return dtype_map.get(dtype, None)


def should_enable_sqmma(a_dtype, b_dtype, M, N, K):
    return (
        (os.getenv("MUSA_ENABLE_SQMMA", "0") == "1")
        and (a_dtype in [torch.float16, torch.bfloat16] and a_dtype.itemsize == 2)
        and ((M, N, K) not in [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)])
    )
