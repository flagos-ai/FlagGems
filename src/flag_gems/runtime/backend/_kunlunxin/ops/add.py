import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def add(A, B, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADD")
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
            out_real = add_func(Ar, Br, alpha)
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
            out_real = add_func(Ar, Br, alpha)
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
            out_real = add_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real.contiguous()).to(
                torch.result_type(A, B)
            )
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN ADD_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return add_func(A, B, alpha, out0=A)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    # elif isinstance(B, torch.Tensor):
    #     return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
