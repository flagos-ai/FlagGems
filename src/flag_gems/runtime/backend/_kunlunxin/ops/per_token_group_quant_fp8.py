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

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0):
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_row_stride) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = y / y_s
    else:
        y_q = (y / _absmax) * fp8_max

    y_q = tl.where(y_q < fp8_min, fp8_min, y_q)
    y_q = tl.where(y_q > fp8_max, fp8_max, y_q).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_v2(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    groups_total,
    groups_per_row,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK_G: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # 2D-tile variant: one program quantizes BLOCK_G consecutive groups so that
    # the per-group absmax reduction is issued once per tile (fewer programs
    # and fewer per-group reduction ops on XPU).
    pid = tl.program_id(0)
    gr = tl.arange(0, BLOCK_G)
    g = pid * BLOCK_G + gr
    ok = g < groups_total
    row = g // groups_per_row
    row_g = g % groups_per_row

    cols = tl.arange(0, BLOCK_S)
    off = row[:, None] * y_row_stride + row_g[:, None] * group_size + cols[None, :]

    y = tl.load(y_ptr + off).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s0 = _absmax / fp8_max
    # NOTE: no python branch here; an `if scale_ue8m0:` branch around the
    # exp2/ceil/log2 row op is miscompiled by XPU TritonXPUCoreTiling, so the
    # ue8m0 scaling is computed unconditionally and selected with tl.where.
    # Also, never build the quantization as `y / (absmax/fp8_max)`: on XPU that
    # divides by subnormals (~1e-38 in fp32-emulation mode) and overflows to
    # inf; keep the safe product form `(y / absmax) * scale` whose quotient is
    # always in (0, 1].
    y_s1 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s0), 1e-10))))
    y_s = tl.where(scale_ue8m0 != 0, y_s1, y_s0)
    recip_ue = tl.exp2(-tl.ceil(tl.log2(tl.maximum(tl.abs(y_s0), 1e-10))))
    y_q_std = (y / _absmax[:, None]) * fp8_max
    y_q_ue = y * recip_ue[:, None]
    y_q = tl.where(scale_ue8m0 != 0, y_q_ue, y_q_std)

    y_q = tl.where(y_q < fp8_min, fp8_min, y_q)
    y_q = tl.where(y_q > fp8_max, fp8_max, y_q).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + off, y_q)
    tl.store(y_s_ptr + g, y_s, mask=ok)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    group_id = g_id % groups_per_row

    y_ptr += row * y_row_stride + group_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += group_id * y_s_col_stride + row

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = y / y_s
    else:
        y_q = (y / _absmax) * fp8_max

    y_q = tl.where(y_q < fp8_min, fp8_min, y_q)
    y_q = tl.where(y_q > fp8_max, fp8_max, y_q).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    M = x.numel() // group_size
    N = group_size

    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        # On XPU, the per-group reduction is issued per program; issuing it once
        # per tile of BLOCK_G groups (v2) cuts the per-group overhead. v2 needs a
        # power-of-two group size whose tile fits in registers, otherwise fall
        # back to the original per-group kernel.
        use_v2 = (N == BLOCK) and N <= 256
        if use_v2:
            # the 2D tile carries BLOCK_G*N lanes per program; make sure at
            # least 4 warps participate (with fewer warps XPU CoreTiling
            # fails and the uni_sram budget overflows)
            _per_token_group_quant_fp8_v2[(triton.cdiv(M, 32),)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                M,
                x.shape[1] // group_size,
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK_G=32,
                BLOCK_S=BLOCK,
                num_warps=max(num_warps, 4),
                num_stages=num_stages,
            )
        else:
            _per_token_group_quant_fp8[(M,)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return x_q, x_s
