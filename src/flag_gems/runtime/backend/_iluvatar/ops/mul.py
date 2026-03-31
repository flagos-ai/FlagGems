import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


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
    """Multiply two tensors element-wise.

    Supports:
    - tensor * tensor
    - tensor * scalar
    - scalar * tensor
    - complex number multiplication
    """
    # Empty tensor protection
    if isinstance(A, torch.Tensor) and A.numel() == 0:
        return torch.empty_like(A)
    if isinstance(B, torch.Tensor) and B.numel() == 0:
        return torch.empty_like(B)

    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )

    if A_is_complex or B_is_complex:
        # 1) A, B both are complex
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
            Br = B.unsqueeze(-1) if isinstance(B, torch.Tensor) else B
            if isinstance(Br, torch.Tensor):
                out_real = mul_func(Ar, Br)
            else:
                out_real = mul_func_scalar(Ar, Br)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
        # 3) A real, B complex
        else:  # not A_is_complex and B_is_complex
            Br = torch.view_as_real(B)
            Ar = A.unsqueeze(-1) if isinstance(A, torch.Tensor) else A
            if isinstance(Ar, torch.Tensor):
                out_real = mul_func(Ar, Br)
            else:
                out_real = mul_func_scalar(Br, Ar)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
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
    """In-place multiply."""
    if isinstance(A, torch.Tensor) and A.numel() == 0:
        return A

    if isinstance(B, torch.Tensor):
        return mul_func(A, B, out0=A)
    else:
        return mul_func_scalar(A, B, out0=A)
