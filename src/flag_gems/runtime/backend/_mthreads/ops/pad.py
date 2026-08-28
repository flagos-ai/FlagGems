# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.pad import constant_pad_nd as default_constant_pad_nd
from flag_gems.ops.pad import pad as default_pad
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Grid axis limits of the MUSA backend (y/z are capped at 65535).
_GRID_LIMIT = 65535

# The one-pass path is limited to the validated high-rho, short-row FP16/BF16
# region where removing the FillCopy double-write provides a stable benefit.
_INT32_MAX = 2**31 - 1
_RANK3_MIN_RHO = 0.9
_RANK3_MAX_L_IN = 512


@libentry()
@triton.jit(do_not_specialize=["value"])
def _constant_pad_1d_kernel(
    in_ptr,
    out_ptr,
    L_in,
    pad_l,
    out_numel,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    copy_mask = (c >= pad_l) & (c < pad_l + L_in) & (c < out_numel)
    x = tl.load(in_ptr + c - pad_l, mask=copy_mask, other=value)
    tl.store(out_ptr + c, x, mask=c < out_numel)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _constant_pad_flat_kernel(
    in_ptr,
    out_ptr,
    L_in,
    L_out,
    row_start,
    row_end,
    pad_l,
    out_numel,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = offset // L_out
    c = offset - row * L_out
    copy_mask = (
        (row >= row_start)
        & (row < row_end)
        & (c >= pad_l)
        & (c < pad_l + L_in)
        & (offset < out_numel)
    )
    src_offset = (row - row_start) * L_in + (c - pad_l)
    x = tl.load(in_ptr + src_offset, mask=copy_mask, other=value)
    tl.store(out_ptr + offset, x, mask=offset < out_numel)


@triton.jit(do_not_specialize=["value"])
def _constant_pad_rank3_onepass_kernel(
    in_ptr,
    out_ptr,
    L_in,
    L_out,
    dim_y_out,
    y_pb,
    y_in,
    outer_pb,
    outer_in,
    pad_l,
    out_numel,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    """Rank-3 one-pass pad with direct outer/y row mapping."""
    pid_y = tl.program_id(0)
    pid_col = tl.program_id(1)
    pid_outer = tl.program_id(2)
    row_valid = (
        (pid_y >= y_pb)
        & (pid_y < y_pb + y_in)
        & (pid_outer >= outer_pb)
        & (pid_outer < outer_pb + outer_in)
    )
    src_row = (pid_outer - outer_pb) * y_in + (pid_y - y_pb)
    c = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_valid = (c >= pad_l) & (c < pad_l + L_in) & (c < L_out)
    out_row = pid_outer * dim_y_out + pid_y
    out_offset = out_row * L_out + c
    copy_mask = row_valid & col_valid & (out_offset < out_numel)
    src_offset = src_row * L_in + (c - pad_l)
    x = tl.load(in_ptr + src_offset, mask=copy_mask, other=value)
    tl.store(out_ptr + out_offset, x, mask=(out_offset < out_numel) & (c < L_out))


@libentry()
@triton.jit(do_not_specialize=["value"])
def _constant_pad_nd_kernel(
    in_ptr,
    out_ptr,
    L_in,
    L_out,
    dim_y,
    y_pb,
    y_sh,
    y_stride,
    m2,
    pb2,
    sh2,
    st2,
    m1,
    pb1,
    sh1,
    st1,
    m0,
    pb0,
    sh0,
    st0,
    pad_l,
    out_numel,
    value,
    SPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    row_valid = (pid_x >= y_pb) & (pid_x < y_pb + y_sh)
    src_row_base = (pid_x - y_pb) * y_stride
    remaining = pid_z
    if SPLIT >= 1:
        i_2 = remaining % m2
        remaining = remaining // m2
        row_valid = row_valid & ((i_2 >= pb2) & (i_2 < pb2 + sh2))
        src_row_base += (i_2 - pb2) * st2
    if SPLIT >= 2:
        i_1 = remaining % m1
        remaining = remaining // m1
        row_valid = row_valid & ((i_1 >= pb1) & (i_1 < pb1 + sh1))
        src_row_base += (i_1 - pb1) * st1
    if SPLIT >= 3:
        i_0 = remaining % m0
        remaining = remaining // m0
        row_valid = row_valid & ((i_0 >= pb0) & (i_0 < pb0 + sh0))
        src_row_base += (i_0 - pb0) * st0
    c = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_valid = (c >= pad_l) & (c < pad_l + L_in) & (c < L_out)
    out_row = pid_z * dim_y + pid_x
    out_offset = out_row * L_out + c
    copy_mask = row_valid & col_valid & (out_offset < out_numel)
    src_offset = src_row_base * L_in + (c - pad_l)
    x = tl.load(in_ptr + src_offset, mask=copy_mask, other=value)
    tl.store(out_ptr + out_offset, x, mask=(out_offset < out_numel) & (c < L_out))


@libentry()
@triton.jit(do_not_specialize=["value"])
def _constant_pad_fill_kernel(
    out_ptr,
    out_numel,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + off, value, mask=off < out_numel)


@libentry()
@triton.jit
def _constant_pad_copy_kernel(
    in_ptr,
    out_ptr,
    L_in,
    L_out,
    dim_y_in,
    y_pb,
    y_dst,
    m2,
    pb2,
    dst2,
    m1,
    pb1,
    dst1,
    m0,
    pb0,
    dst0,
    pad_l,
    SPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    c = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_mask = c < L_in
    in_off = (pid_z * dim_y_in + pid_y) * L_in + c
    out_row_base = (pid_y + y_pb) * y_dst
    remaining = pid_z
    if SPLIT >= 1:
        i_2 = remaining % m2
        remaining = remaining // m2
        out_row_base += (i_2 + pb2) * dst2
    if SPLIT >= 2:
        i_1 = remaining % m1
        remaining = remaining // m1
        out_row_base += (i_1 + pb1) * dst1
    if SPLIT >= 3:
        i_0 = remaining % m0
        remaining = remaining // m0
        out_row_base += (i_0 + pb0) * dst0
    out_off = out_row_base * L_out + c + pad_l
    x = tl.load(in_ptr + in_off, mask=c_mask, other=0.0)
    tl.store(out_ptr + out_off, x, mask=c_mask)


def _parse_pad(ndim, pad):
    pad_before = [0] * ndim
    pad_after = [0] * ndim
    for i in range(len(pad) // 2):
        pad_before[ndim - i - 1] = pad[2 * i]
        pad_after[ndim - i - 1] = pad[2 * i + 1]
    return pad_before, pad_after


def _can_use_triton_kernel(x, pad) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa":
        return False
    if x.ndim == 0 or x.numel() == 0:
        return False
    if not x.is_contiguous():
        return False
    if len(pad) == 0:
        return False
    return True


def _row_space_contiguous(ndim, pad_before, pad_after) -> bool:
    if ndim <= 2:
        return True
    for i in range(1, ndim - 1):
        if pad_before[i] != 0 or pad_after[i] != 0:
            return False
    return True


def _rank3_offsets_fit_int32(out_numel):
    return 0 < out_numel <= _INT32_MAX


def _rank3_onepass_metadata(x, dst_shape, pad_before, pad_after, out_numel):
    L_in = x.shape[-1]
    L_out = dst_shape[-1]
    dim_y_in = x.shape[-2]
    dim_y_out = dst_shape[-2]
    outer_in = x.shape[0]
    outer_out = dst_shape[0]
    row_count = dim_y_out * outer_out
    input_bytes = x.numel() * x.element_size()
    output_bytes = out_numel * x.element_size()
    rho = input_bytes / output_bytes if output_bytes else 0.0
    block = _dtype_block(x, out_numel, 512, 1024)
    grid = (dim_y_out, triton.cdiv(L_out, block), outer_out)
    legal = (
        x.ndim == 3
        and x.dtype in (torch.float16, torch.bfloat16)
        and not _row_space_contiguous(3, pad_before, pad_after)
        and rho >= _RANK3_MIN_RHO
        and L_in <= _RANK3_MAX_L_IN
        and _rank3_offsets_fit_int32(out_numel)
        and dim_y_out > 0
        and outer_out > 0
        and row_count <= _INT32_MAX
        and grid[1] <= _GRID_LIMIT
        and grid[2] <= _GRID_LIMIT
    )
    return {
        "L_in": L_in,
        "L_out": L_out,
        "dim_y_in": dim_y_in,
        "dim_y_out": dim_y_out,
        "outer_in": outer_in,
        "outer_out": outer_out,
        "pad_y": pad_before[-2],
        "pad_outer": pad_before[0],
        "pad_l": pad_before[-1],
        "block": block,
        "grid": grid,
        "rho": rho,
        "legal": legal,
    }


def _should_use_rank3_onepass(x, dst_shape, pad_before, pad_after, out_numel):
    metadata = _rank3_onepass_metadata(x, dst_shape, pad_before, pad_after, out_numel)
    return metadata if metadata["legal"] else None


def _pad_1d(x, pad, value):
    dst_shape, pad_before, pad_after = _dst_shape(x.shape, pad)
    out = torch.empty(dst_shape, device=x.device, dtype=x.dtype)
    L_in = x.shape[0]
    pad_l = pad_before[0]
    out_numel = out.numel()
    block = _dtype_block(x, out_numel, 256, 512)
    grid = (triton.cdiv(out_numel, block),)
    with torch_device_fn.device(x.device):
        _constant_pad_1d_kernel[grid](
            x,
            out,
            L_in,
            pad_l,
            out_numel,
            float(value),
            BLOCK_SIZE=block,
        )
    return out


def _pad_flat(x, pad, value):
    dst_shape, pad_before, pad_after = _dst_shape(x.shape, pad)
    out = torch.empty(dst_shape, device=x.device, dtype=x.dtype)
    ndim = x.ndim
    L_in = x.shape[-1]
    L_out = dst_shape[-1]
    out_numel = out.numel()
    pad_l = pad_before[-1]
    row_start = 0
    for i in range(ndim - 1):
        tail = 1
        for j in range(i + 1, ndim - 1):
            tail *= dst_shape[j]
        row_start += pad_before[i] * tail
    r_in = 1
    for i in range(ndim - 1):
        r_in *= x.shape[i]
    row_end = row_start + r_in
    block = _pick_block(out_numel, 512)
    grid = (triton.cdiv(out_numel, block),)
    with torch_device_fn.device(x.device):
        _constant_pad_flat_kernel[grid](
            x,
            out,
            L_in,
            L_out,
            row_start,
            row_end,
            pad_l,
            out_numel,
            float(value),
            BLOCK_SIZE=block,
        )
    return out


def _pad_v2s(x, pad, value):
    dst_shape, pad_before, pad_after = _dst_shape(x.shape, pad)
    out = torch.empty(dst_shape, device=x.device, dtype=x.dtype)
    ndim = x.ndim
    L_in = x.shape[-1]
    L_out = dst_shape[-1]
    out_numel = out.numel()
    pad_l = pad_before[-1]
    block = _dtype_block(x, out_numel, 512, 1024)
    dim_y = dst_shape[-2]
    z_size = 1
    for i in range(ndim - 2):
        z_size *= dst_shape[i]
    ncol = triton.cdiv(L_out, block)
    if ncol > _GRID_LIMIT or z_size > _GRID_LIMIT or dim_y > 2**31 - 1:
        return None
    split = ndim - 2
    ms = [1, 1, 1]
    pbs = [0, 0, 0]
    shs = [1, 1, 1]
    sts = [0, 0, 0]
    for k, d in enumerate(reversed(range(ndim - 2))):
        if k >= 3:
            return None
        tail = 1
        for j in range(d + 1, ndim - 1):
            tail *= x.shape[j]
        ms[2 - k] = dst_shape[d]
        pbs[2 - k] = pad_before[d]
        shs[2 - k] = x.shape[d]
        sts[2 - k] = tail
    y_stride = 1
    grid = (dim_y, ncol, z_size)
    with torch_device_fn.device(x.device):
        _constant_pad_nd_kernel[grid](
            x,
            out,
            L_in,
            L_out,
            dim_y,
            pad_before[-2],
            x.shape[-2],
            y_stride,
            ms[2],
            pbs[2],
            shs[2],
            sts[2],
            ms[1],
            pbs[1],
            shs[1],
            sts[1],
            ms[0],
            pbs[0],
            shs[0],
            sts[0],
            pad_l,
            out_numel,
            float(value),
            SPLIT=split,
            BLOCK_SIZE=block,
        )
    return out


def _pad_fillcopy(x, pad, value):
    dst_shape, pad_before, pad_after = _dst_shape(x.shape, pad)
    ndim = x.ndim
    L_in = x.shape[-1]
    L_out = dst_shape[-1]
    out_numel = math.prod(dst_shape)
    pad_l = pad_before[-1]
    block = _dtype_block(x, out_numel, 512, 1024)
    split = ndim - 2
    # All fallback conditions must be resolved before output allocation or fill.
    if split < 0 or split > 3:
        return None
    ms = [1, 1, 1]
    pbs = [0, 0, 0]
    dsts = [1, 1, 1]
    for k, d in enumerate(reversed(range(ndim - 2))):
        tail = 1
        for j in range(d + 1, ndim - 1):
            tail *= dst_shape[j]
        ms[2 - k] = x.shape[d]
        pbs[2 - k] = pad_before[d]
        dsts[2 - k] = tail
    dim_y_in = x.shape[-2]
    ncol = triton.cdiv(L_in, block)
    z_size = 1
    for i in range(ndim - 2):
        z_size *= x.shape[i]
    if ncol > _GRID_LIMIT or z_size > _GRID_LIMIT or dim_y_in > 2**31 - 1:
        return None
    rank3 = None
    # Keep the legacy FillCopy launch path free of candidate-gate arithmetic
    # for ranks, dtypes, and widths that can never select the specialized path.
    if (
        ndim == 3
        and x.dtype in (torch.float16, torch.bfloat16)
        and not _row_space_contiguous(3, pad_before, pad_after)
        and L_in <= _RANK3_MAX_L_IN
    ):
        rank3 = _should_use_rank3_onepass(
            x, dst_shape, pad_before, pad_after, out_numel
        )
    out = torch.empty(dst_shape, device=x.device, dtype=x.dtype)
    if rank3 is not None:
        with torch_device_fn.device(x.device):
            _constant_pad_rank3_onepass_kernel[rank3["grid"]](
                x,
                out,
                rank3["L_in"],
                rank3["L_out"],
                rank3["dim_y_out"],
                rank3["pad_y"],
                rank3["dim_y_in"],
                rank3["pad_outer"],
                rank3["outer_in"],
                rank3["pad_l"],
                out_numel,
                float(value),
                BLOCK_SIZE=rank3["block"],
            )
        return out
    with torch_device_fn.device(x.device):
        _constant_pad_fill_kernel[(triton.cdiv(out_numel, block),)](
            out,
            out_numel,
            float(value),
            BLOCK_SIZE=block,
        )
    grid = (ncol, dim_y_in, z_size)
    with torch_device_fn.device(x.device):
        _constant_pad_copy_kernel[grid](
            x,
            out,
            L_in,
            L_out,
            dim_y_in,
            pad_before[-2],
            1,
            ms[2],
            pbs[2],
            dsts[2],
            ms[1],
            pbs[1],
            dsts[1],
            ms[0],
            pbs[0],
            dsts[0],
            pad_l,
            SPLIT=split,
            BLOCK_SIZE=block,
        )
    return out


def _dst_shape(shape, pad):
    ndim = len(shape)
    pad_before, pad_after = _parse_pad(ndim, pad)
    dst_shape = [shape[i] + pad_before[i] + pad_after[i] for i in range(ndim)]
    return dst_shape, pad_before, pad_after


def _pick_block(out_numel, fallback):
    block = fallback
    if out_numel < block:
        block = triton.next_power_of_2(out_numel)
    return block


def _dtype_block(x, out_numel, small, large):
    if x.dtype.itemsize <= 2:
        return _pick_block(out_numel, large)
    return _pick_block(out_numel, small)


def constant_pad_nd(x, pad, value=0):
    logger.debug("GEMS_MTHREADS CONSTANT_PAD_ND")

    if not _can_use_triton_kernel(x, pad):
        return default_constant_pad_nd(x, pad, value)

    ndim = x.ndim
    dst_shape, pad_before, pad_after = _dst_shape(x.shape, pad)

    # Cropping pads are not covered by the MThreads kernels' copy geometry.
    # Preserve ATen semantics rather than entering a partially valid fast path.
    if any(amount < 0 for amount in pad):
        return default_constant_pad_nd(x, pad, value)

    out = None
    if ndim == 1:
        out = _pad_1d(x, pad, value)
    elif ndim == 2:
        if dst_shape[-1] > 512:
            out = _pad_v2s(x, pad, value)
        else:
            out = _pad_flat(x, pad, value)
    elif _row_space_contiguous(ndim, pad_before, pad_after):
        out = _pad_flat(x, pad, value)
    else:
        out = _pad_fillcopy(x, pad, value)

    if out is None:
        return default_constant_pad_nd(x, pad, value)
    return out


def pad(x, pad, mode="constant", value=None):
    logger.debug("GEMS_MTHREADS PAD")

    if mode == "constant":
        if value is None:
            value = 0
        return constant_pad_nd(x, pad, value)
    return default_pad(x, pad, mode, value)
