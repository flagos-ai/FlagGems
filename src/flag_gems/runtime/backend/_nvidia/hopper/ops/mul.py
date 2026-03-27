import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func(x, y):
    return x * y


@triton.jit
def mul_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def mul_all_real_func(x: torch.Tensor, y: torch.Tensor):
    # # We need to preallocate the output.
    # print("\n.......test for mutibackend specific add........\n")
    output = torch.empty_like(x)
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    with torch_device_fn.device(x.device):
        mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch_device_fn.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


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

            # real_out = torch.empty_like(ar, dtype=common_dtype)
            # imag_out = torch.empty_like(ar, dtype=common_dtype)
            shape = ar.shape
            out_buffer = torch.empty((*shape, 2), dtype=common_dtype, device=ar.device)
            real_out = out_buffer[..., 0]
            imag_out = out_buffer[..., 1]
            mul_complex_kernel(ar, ai, br, bi, out0=real_out, out1=imag_out)

            # out = torch.view_as_complex(torch.stack((real_out, imag_out), dim=-1))
            out = torch.view_as_complex(out_buffer)
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
                out_real = mul_func(Ar, Br)  # shape broadcasting requires Ar and Br
            else:
                out_real = mul_func_scalar(Br, Ar)  # Br is tensor, Ar is scalar
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return mul_all_real_func(A, B)
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
