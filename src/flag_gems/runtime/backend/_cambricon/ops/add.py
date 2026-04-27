import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(
    is_tensor=[True, True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func(x, y, alpha, inplace):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha, inplace):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha, inplace):
    return x + y * alpha


def add(A, B, *, alpha=1):
    logger.debug("GEMS_CAMBRICON ADD")
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
            out_real = add_func(Ar, Br, alpha, False)
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
            out_real = add_func(Ar, Br, alpha, False)
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
            out_real = add_func(Ar, Br, alpha, False)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_func(A, B, alpha, False)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, False)
    elif isinstance(B, torch.Tensor):
        return add_func_scalar_tensor(A, B, alpha, False)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1):
    logger.debug("GEMS_CAMBRICON ADD_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_func(A, B, alpha, True, out0=A)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, True, out0=A)
    # elif isinstance(B, torch.Tensor):
    #     return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
