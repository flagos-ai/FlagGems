import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


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
    logger.debug("GEMS MUL")
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


def mul_(A, B):
    logger.debug("GEMS MUL_")
    if isinstance(B, torch.Tensor):
        return mul_func(A, B, out0=A)
    else:
        return mul_func_scalar(A, B, out0=A)
