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

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# sub is bandwidth bound, so the win is in the launcher's tile width, whether
# loads are issued asynchronously, and how much arithmetic the body does.
# Measured on P800 at 4096x4096 (the shipped default config scored fp16 0.45,
# fp32 0.77, bf16 0.26 against torch):
#
# 1. isCloseMemoryAsync=False -- on a load/load/store kernel the async-copy
#    issue overhead costs more than the overlap it buys.
#
# 2. Tile 24576.  The launcher derives the tile as
#    cdiv(buffer_size_limit * 64, element_size) capped at 32768, then rewrites
#    an exactly-32768 tile back down to 16384 to avoid a device hang
#    (_BAD_TILE_SIZE_1D in pointwise_dynamic).  Asking for the full 32768 is
#    therefore counterproductive, while a buffer that lands on 24576 keeps the
#    wider tile.  The buffer needed depends on element size, hence a pair of
#    configs: 768 for 2-byte elements, 1536 for 4-byte.
#
# 3. Dropping the `* alpha` multiply when alpha == 1 is worth 16% on fp16
#    (0.083 -> 0.071 ms).  alpha arrives as an fp32 scalar, so the multiply
#    drags a dtype conversion in with it; fp32 and bf16 measured flat.
#
# 4. kunlunAutoGrid keeps a small tensor on a single cluster, which is what the
#    64x64 shapes want, but its single-cluster grid reads out of bounds on
#    inputs it was not validated against.  It is therefore used only on the
#    plain contiguous same-shape tensor-tensor path (the shape `maximum` also
#    stress-checked); every other form keeps the shipped 12-cluster grid.
#
# Each (tile, grid) pick needs its own kernel: the generated-code file is named
# after the scalar function, so these have to be distinct functions rather than
# one body instantiated repeatedly -- reusing a body under two signatures makes
# the variants collide on a single cache file.  Leave vectorization alone,
# isCloseVectorization=True measured far slower on every pointwise op here.
_BASE_CFG = dict(
    max_tile_size=512,
    max_grid_size=(65536, 65536, 65536),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
)

# buffer 768 -> tile 24576 for 2-byte elements, 1536 -> 24576 for 4-byte.
config_narrow_ = CodeGenConfig(buffer_size_limit=768, **_BASE_CFG)
config_wide_ = CodeGenConfig(buffer_size_limit=1536, **_BASE_CFG)
config_narrow_grid_ = CodeGenConfig(
    buffer_size_limit=768, kunlunAutoGrid=True, **_BASE_CFG
)
config_wide_grid_ = CodeGenConfig(
    buffer_size_limit=1536, kunlunAutoGrid=True, **_BASE_CFG
)

_TT = dict(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
_TTA = dict(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
_TS = dict(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
_TSA = dict(is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")])
_ST = dict(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
_STA = dict(is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")])


# --- tensor - tensor, alpha == 1 (no multiply) -------------------------------
@pointwise_dynamic(**_TT, config=config_narrow_)
@triton.jit
def _sub_tt_narrow(x, y):
    return x - y


@pointwise_dynamic(**_TT, config=config_wide_)
@triton.jit
def _sub_tt_wide(x, y):
    return x - y


@pointwise_dynamic(**_TT, config=config_narrow_grid_)
@triton.jit
def _sub_tt_narrow_grid(x, y):
    return x - y


@pointwise_dynamic(**_TT, config=config_wide_grid_)
@triton.jit
def _sub_tt_wide_grid(x, y):
    return x - y


# --- tensor - tensor * alpha -------------------------------------------------
@pointwise_dynamic(**_TTA, config=config_narrow_)
@triton.jit
def _sub_tta_narrow(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(**_TTA, config=config_wide_)
@triton.jit
def _sub_tta_wide(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(**_TTA, config=config_narrow_grid_)
@triton.jit
def _sub_tta_narrow_grid(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(**_TTA, config=config_wide_grid_)
@triton.jit
def _sub_tta_wide_grid(x, y, alpha):
    return x - y * alpha


# --- tensor - scalar ---------------------------------------------------------
@pointwise_dynamic(**_TS, config=config_narrow_)
@triton.jit
def _sub_ts_narrow(x, y):
    return x - y


@pointwise_dynamic(**_TS, config=config_wide_)
@triton.jit
def _sub_ts_wide(x, y):
    return x - y


@pointwise_dynamic(**_TSA, config=config_narrow_)
@triton.jit
def _sub_tsa_narrow(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(**_TSA, config=config_wide_)
@triton.jit
def _sub_tsa_wide(x, y, alpha):
    return x - y * alpha


# --- scalar - tensor ---------------------------------------------------------
@pointwise_dynamic(**_ST, config=config_narrow_)
@triton.jit
def _sub_st_narrow(x, y):
    return x - y


@pointwise_dynamic(**_ST, config=config_wide_)
@triton.jit
def _sub_st_wide(x, y):
    return x - y


@pointwise_dynamic(**_STA, config=config_narrow_)
@triton.jit
def _sub_sta_narrow(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(**_STA, config=config_wide_)
@triton.jit
def _sub_sta_wide(x, y, alpha):
    return x - y * alpha


def _is_unit(alpha):
    # Only a plain real 1 lets us drop the multiply; a complex alpha or a
    # 0-d tensor keeps the general kernel.
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        return False
    return alpha == 1


def sub_func(x, y, alpha, **kwargs):
    # The output dtype drives the derived tile, so promote first.
    wide = torch.promote_types(x.dtype, y.dtype).itemsize > 2
    out = kwargs.get("out0")
    # autoGrid only on the validated shape: same-shape contiguous operands,
    # writing to a contiguous destination of that same shape.
    grid = (
        x.shape == y.shape
        and x.is_contiguous()
        and y.is_contiguous()
        and (out is None or (out.is_contiguous() and out.shape == x.shape))
    )
    if _is_unit(alpha):
        if grid:
            kern = _sub_tt_wide_grid if wide else _sub_tt_narrow_grid
        else:
            kern = _sub_tt_wide if wide else _sub_tt_narrow
        return kern(x, y, **kwargs)
    if grid:
        kern = _sub_tta_wide_grid if wide else _sub_tta_narrow_grid
    else:
        kern = _sub_tta_wide if wide else _sub_tta_narrow
    return kern(x, y, alpha, **kwargs)


def sub_func_tensor_scalar(x, y, alpha, **kwargs):
    wide = torch.result_type(x, y).itemsize > 2
    if _is_unit(alpha):
        kern = _sub_ts_wide if wide else _sub_ts_narrow
        return kern(x, y, **kwargs)
    kern = _sub_tsa_wide if wide else _sub_tsa_narrow
    return kern(x, y, alpha, **kwargs)


def sub_func_scalar_tensor(x, y, alpha, **kwargs):
    wide = torch.result_type(y, x).itemsize > 2
    if _is_unit(alpha):
        kern = _sub_st_wide if wide else _sub_st_narrow
        return kern(x, y, **kwargs)
    kern = _sub_sta_wide if wide else _sub_sta_narrow
    return kern(x, y, alpha, **kwargs)


def sub(A, B, *, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN SUB")
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        if A_is_complex and B_is_complex:
            Ar = torch.view_as_real(A)
            Br = torch.view_as_real(B)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = sub_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
        elif A_is_complex and not B_is_complex:
            Ar = torch.view_as_real(A)
            if isinstance(B, torch.Tensor):
                B_casted = B.to(dtype=Ar.dtype)
                Br = torch.stack([B_casted, torch.zeros_like(B_casted)], dim=-1)
            else:
                B_tensor = torch.full_like(Ar[..., 0], fill_value=B, dtype=Ar.dtype)
                Br = torch.stack([B_tensor, torch.zeros_like(B_tensor)], dim=-1)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = sub_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real.contiguous()).to(
                torch.result_type(A, B)
            )
        else:
            Br = torch.view_as_real(B)
            if isinstance(A, torch.Tensor):
                A_casted = A.to(dtype=Br.dtype)
                Ar = torch.stack([A_casted, torch.zeros_like(A_casted)], dim=-1)
            else:
                A_tensor = torch.full_like(Br[..., 0], fill_value=A, dtype=Br.dtype)
                Ar = torch.stack([A_tensor, torch.zeros_like(A_tensor)], dim=-1)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = sub_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real.contiguous()).to(
                torch.result_type(A, B)
            )
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return sub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return sub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return sub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN SUB_")
    if isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha, out0=A)
    else:
        return sub_func_tensor_scalar(A, B, alpha, out0=A)


def subtract_(A, B, *, alpha=1):
    """In-place subtraction: A.subtract_(B, alpha) == A -= B * alpha."""
    logger.debug("GEMS_KUNLUNXIN SUBTRACT_")
    return sub_(A, B, alpha=alpha)
