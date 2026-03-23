import logging

import torch
import triton

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
            ar, ai = ar.to(common_dtype), ai.to(common_dtype)
            br, bi = br.to(common_dtype), bi.to(common_dtype)

            real_out = torch.empty_like(ar, dtype=common_dtype)
            imag_out = torch.empty_like(ar, dtype=common_dtype)
            mul_complex_kernel(ar, ai, br, bi, out0=real_out, out1=imag_out)

            out = torch.view_as_complex(torch.stack((real_out, imag_out), dim=-1))
            return out.to(torch.result_type(A, B))
        # 2) A complex, B real
        elif A_is_complex and not B_is_complex:
            Ar = torch.view_as_real(A)
            ar, ai = Ar[..., 0], Ar[..., 1]
            br = B
            common_dtype = torch.promote_types(ar.dtype, br.dtype)
            ar, ai = ar.to(common_dtype), ai.to(common_dtype)
            br = br.to(common_dtype)
            bi = torch.zeros_like(br, dtype=common_dtype)

            real_out = torch.empty_like(ar, dtype=common_dtype)
            imag_out = torch.empty_like(ar, dtype=common_dtype)
            mul_complex_kernel(ar, ai, br, bi, out0=real_out, out1=imag_out)

            out = torch.view_as_complex(torch.stack((real_out, imag_out), dim=-1))
            return out.to(torch.result_type(A, B))
        # 3) A real, B complex
        else:  # not A_is_complex and B_is_complex
            Br = torch.view_as_real(B)
            br, bi = Br[..., 0], Br[..., 1]
            ar = A
            common_dtype = torch.promote_types(ar.dtype, br.dtype)
            ar = ar.to(common_dtype)
            ai = torch.zeros_like(ar, dtype=common_dtype)
            br, bi = br.to(common_dtype), bi.to(common_dtype)

            real_out = torch.empty_like(br, dtype=common_dtype)
            imag_out = torch.empty_like(br, dtype=common_dtype)
            mul_complex_kernel(ar, ai, br, bi, out0=real_out, out1=imag_out)

            out = torch.view_as_complex(torch.stack((real_out, imag_out), dim=-1))
            return out.to(torch.result_type(A, B))
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
