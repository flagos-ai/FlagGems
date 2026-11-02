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

from flag_gems.runtime import torch_device_fn
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func(x, y):
    return x * y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func_scalar(x, y):
    return x * y


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, 2, 3, "DEFAULT"), (0, 1, 2, 3, "DEFAULT")],
)
@triton.jit
def mul_complex_kernel(ar, ai, br, bi):
    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    return real, imag


def mul(A, B):
    logger.debug("GEMS_KUNLUNXIN MUL")
    A_is_complex = isinstance(A, torch.Tensor) and A.is_complex()
    B_is_complex = isinstance(B, torch.Tensor) and B.is_complex()

    if A_is_complex or B_is_complex:
        if A_is_complex and B_is_complex:
            Ar = torch.view_as_real(A.resolve_conj())
            Br = torch.view_as_real(B.resolve_conj())
            ar, ai = Ar[..., 0].contiguous(), Ar[..., 1].contiguous()
            br, bi = Br[..., 0].contiguous(), Br[..., 1].contiguous()
            # Upcast float16 to float32 to avoid precision loss in
            # complex multiplication (ac-bd, ad+bc)
            orig_dtype = ar.dtype
            if orig_dtype == torch.float16:
                ar, ai = ar.to(torch.float32), ai.to(torch.float32)
                br, bi = br.to(torch.float32), bi.to(torch.float32)
            real_out, imag_out = mul_complex_kernel(ar, ai, br, bi)
            if orig_dtype == torch.float16:
                real_out = real_out.to(orig_dtype)
                imag_out = imag_out.to(orig_dtype)
            out = torch.view_as_complex(torch.stack((real_out, imag_out), dim=-1))
            return out.to(torch.result_type(A, B))
        elif A_is_complex and not B_is_complex:
            Ar = torch.view_as_real(A.resolve_conj())
            if isinstance(B, torch.Tensor):
                Br = B.unsqueeze(-1)
                out_real = mul_func(Ar, Br)
            else:
                out_real = mul_func_scalar(Ar, B)
            return torch.view_as_complex(out_real.contiguous())
        else:
            Br = torch.view_as_real(B.resolve_conj())
            if isinstance(A, torch.Tensor):
                Ar = A.unsqueeze(-1)
                out_real = mul_func(Ar, Br)
            else:
                out_real = mul_func_scalar(Br, A)
            return torch.view_as_complex(out_real.contiguous())

    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return mul_func(A, B)
    elif isinstance(A, torch.Tensor):
        return mul_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return mul_func_scalar(B, A)
    else:
        # Both scalar
        return torch.tensor(A * B)


# Wide tiles measured fastest for this streaming pattern on XPU: 4096 lanes
# ran at 306 GB/s, 16384 at 593 GB/s.
_BLOCK = 16384

# Upper bound on the elements this kernel is used for.  Below it the launch is
# host-bound -- a fused launch costs ~26 us against ~73 us for the generated
# one -- and the fused kernel wins by 1.6-2x.  Above it `pointwise_dynamic`'s
# generated kernel is the faster of the two: it vectorises to 128-bit accesses
# and reaches 1.2-1.6 TB/s on this part where the plain 16384-lane form tops
# out at ~590 GB/s (measured on 4x8x64^3-shaped and 16.7M-element tensors).
_FUSED_MAX_ELEMS = 1 << 20


@triton.jit
def _mul_inplace_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    """``a *= b`` for the exact case: same dtype, same shape, both contiguous.

    ``pointwise_dynamic`` has to keep the general index mapping, the promotion
    table and the out-buffer plumbing on every call even when none of it is
    needed, which costs ~50 us of host time per launch.  That dominates every
    tensor small enough to fit in a handful of tiles.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(a_ptr + offs, a * b, mask=mask)


def _mul_inplace_fast(A, B):
    """Launch the fused kernel when it is exactly equivalent, else return None."""
    if (
        not isinstance(B, torch.Tensor)
        or A.shape != B.shape
        or A.dtype is not B.dtype
        or A.device != B.device
        or A.dtype == torch.bool
        or A.is_complex()
        or not A.is_contiguous()
        or not B.is_contiguous()
    ):
        return None
    n = A.numel()
    if n == 0:
        return A
    if n > _FUSED_MAX_ELEMS:
        return None
    grid = (triton.cdiv(n, _BLOCK),)
    with torch_device_fn.device(A.device):
        _mul_inplace_kernel[grid](A, B, n, BLOCK=_BLOCK, num_warps=4)
    return A


def mul_(A, B):
    logger.debug("GEMS_KUNLUNXIN MUL_")
    fast = _mul_inplace_fast(A, B)
    if fast is not None:
        return fast
    if isinstance(B, torch.Tensor):
        return mul_func(A, B, out0=A)
    else:
        return mul_func_scalar(A, B, out0=A)
