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

import torch
import triton
import triton.language as tl

from ..utils.codegen_config_utils import get_pointwise_nd_codegen_config
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_MUL_TENSOR_CONFIG = get_pointwise_nd_codegen_config()

# --- 4.2 fast-path tunings (scalar / row-broadcast) ---
_BLOCK_N = 1024 * 4  # 4096
_TOTAL_CORE_NUM = 16
_MAX_COL_TILE = 2048
# tsingmicro 等后端 f32 整数精度约 2^24，i0*stride_d0 需分块
_F32_INT_MAX = (1 << 24) - 1


def _col_tile_size(n: int, max_tile: int = _BLOCK_N) -> int:
    return min(max_tile, triton.next_power_of_2(max(n, 1)))


def _d0_chunk_size(stride_d0: int) -> int:
    if stride_d0 <= 0:
        return 1 << 30
    return max(1, _F32_INT_MAX // stride_d0)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=_MUL_TENSOR_CONFIG,
)
@triton.jit
def mul_func(x, y, inplace):
    return x * y


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def mul_func_scalar(x, y, inplace):
    return x * y


@pointwise_dynamic(
    is_tensor=[True, True, True, True],  # ar, ai, br, bi
    num_outputs=2,
    promotion_methods=[(0, 1, 2, 3, "DEFAULT"), (0, 1, 2, 3, "DEFAULT")],
)
@triton.jit
def mul_complex_kernel(ar, ai, br, bi):
    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    return real, imag


@triton.jit
def _mul_full_x_scalar_kernel(
    full_ptr,
    scalar_ptr,
    out_ptr,
    n_elements,
    iters,
    BLOCK: tl.constexpr,
):
    """Contiguous full * 1-element tensor: one scalar load per CTA, flat GSL."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    s = tl.load(scalar_ptr)
    n_i64 = tl.cast(n_elements, tl.int64)
    for i in range(iters):
        start = (tl.cast(i, tl.int64) * num_programs + pid) * BLOCK
        offs = start + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n_i64
        v = tl.load(full_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, v * s, mask=mask)


@triton.jit
def _mul_row_broadcast_kernel_2d(
    full_ptr,
    scalar_ptr,
    out_ptr,
    stride_full0,
    stride_full1,
    stride_scalar0,
    stride_scalar1,
    stride_out0,
    stride_out1,
    n_cols,
    num_blocks,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(scalar_ptr + row * stride_scalar0 + 0 * stride_scalar1)
    for blk in range(num_blocks):
        cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        v = tl.load(
            full_ptr + row * stride_full0 + cols * stride_full1,
            mask=mask,
            other=0.0,
        )
        tl.store(
            out_ptr + row * stride_out0 + cols * stride_out1,
            v * s,
            mask=mask,
        )


@triton.jit
def _mul_row_broadcast_kernel_3d(
    full_ptr,
    brc_ptr,
    out_ptr,
    stride_f0,
    stride_f1,
    stride_f2,
    stride_s0,
    stride_s1,
    stride_s2,
    stride_o0,
    stride_o1,
    stride_o2,
    d0,
    d1,
    iters,
    n_cols,
    num_blocks,
    BLOCK_N: tl.constexpr,
):
    """[d0,d1,N]*[d0,d1,1]: grid <=16, GSL rows; brc load once per row."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_rows = d0 * d1
    for i in range(iters):
        row = i * num_programs + pid
        if row < num_rows:
            i0 = row // d1
            i1 = row % d1
            s = tl.load(
                brc_ptr + i0 * stride_s0 + i1 * stride_s1 + 0 * stride_s2,
            )
            base_f = i0 * stride_f0 + i1 * stride_f1
            base_o = i0 * stride_o0 + i1 * stride_o1
            for blk in range(num_blocks):
                cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = cols < n_cols
                v = tl.load(
                    full_ptr + base_f + cols * stride_f2,
                    mask=mask,
                    other=0.0,
                )
                tl.store(
                    out_ptr + base_o + cols * stride_o2,
                    v * s,
                    mask=mask,
                )


@triton.jit
def _mul_broadcast_kernel_4d(
    full_ptr,
    brc_ptr,
    out_ptr,
    stride_f0,
    stride_f1,
    stride_f2,
    stride_f3,
    stride_b0,
    stride_b3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    d2,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    """full[i0,i1,i2,:] * brc[i0,0,0,:] with grid (d0, d1*d2, col_blocks)."""
    i0 = tl.program_id(0)
    row = tl.program_id(1)
    blk = tl.program_id(2)
    i1 = row // d2
    i2 = row % d2
    cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    s = tl.load(
        brc_ptr + i0 * stride_b0 + cols * stride_b3,
        mask=mask,
        other=0.0,
    )
    v = tl.load(
        full_ptr + i0 * stride_f0 + i1 * stride_f1 + i2 * stride_f2 + cols * stride_f3,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + i0 * stride_o0 + i1 * stride_o1 + i2 * stride_o2 + cols * stride_o3,
        v * s,
        mask=mask,
    )


def _pick_mul_broadcast_2d(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]] | None:
    if a.ndim != 2 or b.ndim != 2:
        return None
    if a.device != b.device or a.dtype != b.dtype:
        return None
    if a.shape[0] != b.shape[0]:
        return None
    if a.shape[1] > 1 and b.shape[1] == 1:
        return a, b, tuple(a.shape)
    if a.shape[1] == 1 and b.shape[1] > 1:
        return b, a, tuple(b.shape)
    return None


def _pick_mul_broadcast_3d(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]] | None:
    if a.ndim != 3 or b.ndim != 3:
        return None
    if a.device != b.device or a.dtype != b.dtype:
        return None
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
        return None
    if a.shape[2] > 1 and b.shape[2] == 1:
        return a, b, tuple(a.shape)
    if a.shape[2] == 1 and b.shape[2] > 1:
        return b, a, tuple(b.shape)
    return None


def _pick_mul_broadcast_4d(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]] | None:
    if a.ndim != 4 or b.ndim != 4:
        return None
    if a.device != b.device or a.dtype != b.dtype:
        return None
    if a.shape[0] != b.shape[0] or a.shape[3] != b.shape[3]:
        return None
    if a.shape[1] > 1 and a.shape[2] > 1 and b.shape[1] == 1 and b.shape[2] == 1:
        return a, b, tuple(a.shape)
    if b.shape[1] > 1 and b.shape[2] > 1 and a.shape[1] == 1 and a.shape[2] == 1:
        return b, a, tuple(b.shape)
    return None


def _mul_fast_broadcast_2d(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor | None:
    picked = _pick_mul_broadcast_2d(a, b)
    if picked is None:
        return None
    full, brc, out_shape = picked
    m, n = out_shape
    if out is None:
        out = torch.empty(out_shape, device=full.device, dtype=full.dtype)
    block_n = _BLOCK_N
    num_blocks = triton.cdiv(n, block_n)
    _mul_row_broadcast_kernel_2d[(m,)](
        full,
        brc,
        out,
        full.stride(0),
        full.stride(1),
        brc.stride(0),
        brc.stride(1),
        out.stride(0),
        out.stride(1),
        n,
        num_blocks,
        BLOCK_N=block_n,
    )
    return out


def _mul_launch_3d(full: torch.Tensor, brc: torch.Tensor, out: torch.Tensor) -> None:
    d0, d1, n = out.shape
    block_n = _col_tile_size(n, _MAX_COL_TILE)
    num_blocks = triton.cdiv(n, block_n)
    num_rows = d0 * d1
    iters = triton.cdiv(num_rows, _TOTAL_CORE_NUM)
    grid = (min(num_rows, _TOTAL_CORE_NUM),)
    _mul_row_broadcast_kernel_3d[grid](
        full,
        brc,
        out,
        full.stride(0),
        full.stride(1),
        full.stride(2),
        brc.stride(0),
        brc.stride(1),
        brc.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        d0,
        d1,
        iters,
        n,
        num_blocks,
        BLOCK_N=block_n,
    )


def _mul_fast_broadcast_3d(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor | None:
    picked = _pick_mul_broadcast_3d(a, b)
    if picked is None:
        return None
    full, brc, out_shape = picked
    if out is None:
        out = torch.empty(out_shape, device=full.device, dtype=full.dtype)

    d0 = out_shape[0]
    chunk = _d0_chunk_size(full.stride(0))
    if d0 <= chunk:
        _mul_launch_3d(full, brc, out)
        return out

    for d0_start in range(0, d0, chunk):
        sl = slice(d0_start, min(d0_start + chunk, d0))
        _mul_launch_3d(full[sl], brc[sl], out[sl])
    return out


def _mul_launch_4d(full: torch.Tensor, brc: torch.Tensor, out: torch.Tensor) -> None:
    d0, d1, d2, n = out.shape
    block_n = _col_tile_size(n)
    col_blocks = triton.cdiv(n, block_n)
    _mul_broadcast_kernel_4d[(d0, d1 * d2, col_blocks)](
        full,
        brc,
        out,
        full.stride(0),
        full.stride(1),
        full.stride(2),
        full.stride(3),
        brc.stride(0),
        brc.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        d2,
        n,
        BLOCK_N=block_n,
    )


def _mul_fast_broadcast_4d(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor | None:
    picked = _pick_mul_broadcast_4d(a, b)
    if picked is None:
        return None
    full, brc, out_shape = picked
    if out is None:
        out = torch.empty(out_shape, device=full.device, dtype=full.dtype)

    d0 = out_shape[0]
    chunk = _d0_chunk_size(full.stride(0))
    if d0 <= chunk:
        _mul_launch_4d(full, brc, out)
        return out

    for d0_start in range(0, d0, chunk):
        sl = slice(d0_start, min(d0_start + chunk, d0))
        _mul_launch_4d(full[sl], brc[sl], out[sl])
    return out


def _pick_mul_scalar_tensor(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One side is a 1-element tensor broadcasting as a scalar (e.g. [N,H,W]*[1])."""
    if a.device != b.device or a.dtype != b.dtype:
        return None
    a_one = a.numel() == 1
    b_one = b.numel() == 1
    if a_one == b_one:
        return None
    full, scalar = (b, a) if a_one else (a, b)
    if not full.is_contiguous():
        return None
    return full, scalar.reshape(())


def _mul_fast_scalar_tensor(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor | None:
    picked = _pick_mul_scalar_tensor(a, b)
    if picked is None:
        return None
    full, scalar = picked
    if out is None:
        out = torch.empty(full.shape, device=full.device, dtype=full.dtype)
    elif out.shape != full.shape or not out.is_contiguous():
        return None

    n = full.numel()
    block = _BLOCK_N
    tiles = triton.cdiv(n, block)
    grid = (min(tiles, _TOTAL_CORE_NUM),)
    iters = triton.cdiv(tiles, grid[0])
    _mul_full_x_scalar_kernel[grid](
        full,
        scalar,
        out,
        n,
        iters,
        BLOCK=block,
    )
    return out


def _mul_fast_broadcast(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor | None:
    # Tensor * 1-element tensor (trace often maps scalar RHS as shape [1]).
    fast = _mul_fast_scalar_tensor(a, b, out=out)
    if fast is not None:
        return fast
    # 4d path kept from 4.2 but disabled there; leave commented for parity.
    # if a.ndim == 4 and b.ndim == 4:
    #     return _mul_fast_broadcast_4d(a, b, out=out)
    if a.ndim == 3 and b.ndim == 3:
        return _mul_fast_broadcast_3d(a, b, out=out)
    if a.ndim == 2 and b.ndim == 2:
        return _mul_fast_broadcast_2d(a, b, out=out)
    return None


def mul(A, B):
    logger.debug("GEMS_TSINGMICRO MUL")
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        # 1) A、B both are complex
        if A_is_complex and B_is_complex:
            Ar = torch.view_as_real(A)
            Br = torch.view_as_real(B)
            ar, ai = Ar[..., 0], Ar[..., 1]
            br, bi = Br[..., 0], Br[..., 1]
            common_dtype = torch.promote_types(ar.dtype, br.dtype)
            compute_dtype = (
                torch.float32 if common_dtype == torch.float16 else common_dtype
            )
            ar, ai = ar.to(compute_dtype), ai.to(compute_dtype)
            br, bi = br.to(compute_dtype), bi.to(compute_dtype)

            real_out = torch.empty_like(ar, dtype=compute_dtype)
            imag_out = torch.empty_like(ar, dtype=compute_dtype)
            mul_complex_kernel(ar, ai, br, bi, out0=real_out, out1=imag_out)

            out = torch.view_as_complex(torch.stack((real_out, imag_out), dim=-1))
            return out.to(torch.result_type(A, B))
        # 2) A complex, B real
        elif A_is_complex and not B_is_complex:
            Ar = torch.view_as_real(A)
            Br = B.unsqueeze(-1) if isinstance(B, torch.Tensor) else B
            if isinstance(Br, torch.Tensor):
                out_real = mul_func(Ar, Br, False)
            else:
                out_real = mul_func_scalar(Ar, Br, False)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
        # 3) A real, B complex
        else:  # not A_is_complex and B_is_complex
            Br = torch.view_as_real(B)
            Ar = A.unsqueeze(-1) if isinstance(A, torch.Tensor) else A
            if isinstance(Ar, torch.Tensor):
                out_real = mul_func(
                    Ar, Br, False
                )  # shape broadcasting requires Ar and Br
            else:
                out_real = mul_func_scalar(Br, Ar, False)  # Br is tensor, Ar is scalar
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.device != B.device:
            if A.dim() == 0:
                assert A.device == torch.device("cpu"), "expect scalar tensor on cpu"
                A = A.to(B.device)
            elif B.dim() == 0:
                assert B.device == torch.device("cpu"), "expect scalar tensor on cpu"
                B = B.to(A.device)
        fast = _mul_fast_broadcast(A, B)
        if fast is not None:
            return fast
        return mul_func(A, B, False)
    elif isinstance(A, torch.Tensor):
        return mul_func_scalar(A, B, False)
    elif isinstance(B, torch.Tensor):
        return mul_func_scalar(B, A, False)
    else:
        # Both scalar
        return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS_TSINGMICRO MUL_")
    if isinstance(B, torch.Tensor):
        if B.device != A.device and B.dim() == 0:
            assert B.device == torch.device("cpu"), "expect scalar tensor on cpu"
            B = B.to(A.device)
        fast = _mul_fast_broadcast(A, B, out=A)
        if fast is not None:
            return A
        return mul_func(A, B, True, out0=A)
    else:
        return mul_func_scalar(A, B, True, out0=A)
