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


def mul(A, B):
    logger.debug("GEMS MUL")
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        if A_is_complex and B_is_complex:
            A_r, A_i = A.real, A.imag
            B_r, B_i = B.real, B.imag
            if (
                A_r.dtype == torch.float16
            ):  # complex32's real and imaginary parts are fp16
                A_r, A_i, B_r, B_i = A_r.float(), A_i.float(), B_r.float(), B_i.float()
            ac = mul(A_r, B_r)
            bd = mul(A_i, B_i)
            ad = mul(A_r, B_i)
            bc = mul(A_i, B_r)
            out = torch.complex(ac - bd, ad + bc)
        elif A_is_complex and not B_is_complex:
            A_r, A_i = A.real, A.imag
            if A_r.dtype == torch.float16:
                A_r, A_i = A_r.float(), A_i.float()
            out = torch.complex(mul(A_r, B), mul(A_i, B))
        else:  # not A_is_complex and B_is_complex
            B_r, B_i = B.real, B.imag
            if B_r.dtype == torch.float16:
                B_r, B_i = B_r.float(), B_i.float()
            out = torch.complex(mul(A, B_r), mul(A, B_i))
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
