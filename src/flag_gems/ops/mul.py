import logging

import torch
import triton

from flag_gems.ops.resolve_conj import resolve_conj
from flag_gems.ops.view_as_complex import view_as_complex as fg_view_as_complex
from flag_gems.utils import pointwise_dynamic

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
    is_tensor=[True, True, True, True],  # ar, ai, br, bi
    num_outputs=2,
    promotion_methods=[(0, 1, 2, 3, "DEFAULT"), (0, 1, 2, 3, "DEFAULT")],
)
@triton.jit
def mul_complex_kernel(ar, ai, br, bi):
    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    return real, imag


def mul(A, B):
    logger.debug("GEMS MUL")

    if isinstance(A, torch.Tensor) and A.dtype.is_complex:
        A = resolve_conj(A)
    if isinstance(B, torch.Tensor) and B.dtype.is_complex:
        B = resolve_conj(B)

    A_is_complex = (isinstance(A, torch.Tensor) and A.dtype.is_complex) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.dtype.is_complex) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        # 1) A、B both are complex
        if A_is_complex and B_is_complex:
            ar, ai = A.real, A.imag
            br, bi = B.real, B.imag
            common_dtype = torch.promote_types(
                torch.promote_types(ar.dtype, ai.dtype),
                torch.promote_types(br.dtype, bi.dtype),
            )

            out_real = torch.empty((*ar.shape, 2), dtype=common_dtype, device=ar.device)
            mul_complex_kernel(
                ar,
                ai,
                br,
                bi,
                out0=out_real[..., 0],
                out1=out_real[..., 1],
            )

            out = fg_view_as_complex(out_real)
            return out.to(torch.result_type(A, B))
        # 2) A complex, B real
        elif A_is_complex and not B_is_complex:
            ar, ai = A.real, A.imag
            out_real = torch.empty((*ar.shape, 2), dtype=ar.dtype, device=ar.device)
            if isinstance(B, torch.Tensor):
                mul_func(ar, B, out0=out_real[..., 0])
                mul_func(ai, B, out0=out_real[..., 1])
            else:
                mul_func_scalar(ar, B, out0=out_real[..., 0])
                mul_func_scalar(ai, B, out0=out_real[..., 1])
            return fg_view_as_complex(out_real).to(torch.result_type(A, B))
        # 3) A real, B complex
        else:  # not A_is_complex and B_is_complex
            br, bi = B.real, B.imag
            out_real = torch.empty((*br.shape, 2), dtype=br.dtype, device=br.device)
            if isinstance(A, torch.Tensor):
                mul_func(A, br, out0=out_real[..., 0])
                mul_func(A, bi, out0=out_real[..., 1])
            else:
                mul_func_scalar(br, A, out0=out_real[..., 0])
                mul_func_scalar(bi, A, out0=out_real[..., 1])
            return fg_view_as_complex(out_real).to(torch.result_type(A, B))
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return mul_func(A, B)
    elif isinstance(A, torch.Tensor):
        return mul_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return mul_func_scalar(B, A)
    else:
        # Both scalar
        return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS MUL_")
    if isinstance(B, torch.Tensor):
        return mul_func(A, B, out0=A)
    else:
        return mul_func_scalar(A, B, out0=A)
