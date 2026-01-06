import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils.tensor_wrapper import StridedBuffer

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

my_config = CodeGenConfig(
    max_tile_size= 65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")], config=my_config)
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")], config=my_config
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")], config=my_config
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def add(A, B, *, alpha=1):
    logger.debug("GEMS ADD")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return add_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
# def add(x: torch.Tensor, y: torch.Tensor, *, alpha=1):
#     # output_torch = x + y
#     # x_ = x.to(DEVICE)
#     # y_ = y.to(DEVICE)
#     # We need to preallocate the output.
#     output = torch.empty_like(x)
#     # assert x.is_cuda and y.is_cuda and output.is_cuda
#     n_elements = output.numel()
#     # The SPMD launch grid denotes the number of kernel instances that run in parallel.
#     # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
#     # In this case, we use a 1D grid where the size is the number of blocks:
#     grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
#     # NOTE:
#     #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
#     #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
#     #  - Don't forget to pass meta-parameters as keywords arguments.
#     print("++++++========================:",alpha)
#     add_kernel[grid](x, y, alpha, output, n_elements, BLOCK_SIZE=1024)
#     # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
#     # running asynchronously at this point.
#     output = output.to("cpu")
#     # print(x, " + ", y, " =\n\t", output , ", exp  ", output_torch)
#     # print(
#     #     f"The maximum difference between torch and triton is "
#     #     f"{torch.max(torch.abs(output_torch - output))}"
#     # )
#     return output


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    alpha,
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
    output = x + y * alpha
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def add_(A, B, *, alpha=1):
    logger.debug("GEMS ADD_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return add_func(A, B, alpha, out0=A)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    # elif isinstance(B, torch.Tensor):
    #     return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
