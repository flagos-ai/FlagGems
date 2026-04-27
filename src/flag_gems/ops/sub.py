import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.pointwise_dynamic import ComplexMode

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


# Register complex support (elementwise)
sub_func.register_complex(mode=ComplexMode.ELEMENTWISE)
sub_func_tensor_scalar.register_complex(
    mode=ComplexMode.ELEMENTWISE, tensorize_scalars=True, fallback_target=sub_func
)
sub_func_scalar_tensor.register_complex(
    mode=ComplexMode.ELEMENTWISE, tensorize_scalars=True, fallback_target=sub_func
)


def sub(A, B, *, alpha=1):
    logger.debug("GEMS SUB")
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
                Br = torch.view_as_real(B.to(A.dtype))
            else:
                Br = torch.view_as_real(
                    torch.tensor(B, dtype=A.dtype, device=A.device).expand_as(A)
                )
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = sub_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
        else:
            Br = torch.view_as_real(B)
            if isinstance(A, torch.Tensor):
                Ar = torch.view_as_real(A.to(B.dtype))
            else:
                Ar = torch.view_as_real(
                    torch.tensor(A, dtype=B.dtype, device=B.device).expand_as(B)
                )
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = sub_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return sub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return sub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS SUB_")
    if isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha, out0=A)
    else:
        return sub_func_tensor_scalar(A, B, alpha, out0=A)
