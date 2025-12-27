import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def compute_vdot(
    inp_real, inp_imag, other_real, other_imag, inp_is_conj, other_is_conj
):
    # # Given inp storage: [inp_real, inp_imag], other: [other_real, other_imag]

    # # Case 1: inp_is_conj = False, other_is_conj = False
    # out_real = inp_real * other_real + inp_imag * other_imag
    # out_imag = inp_real * other_imag - inp_imag * other_real

    # # Case 2: inp_is_conj = True, other_is_conj = False
    # out_real = inp_real * other_real - inp_imag * other_imag
    # out_imag = inp_real * other_imag + inp_imag * other_real

    # # Case 3: inp_is_conj = False, other_is_conj = True
    # out_real = inp_real * other_real - inp_imag * other_imag
    # out_imag = -inp_real * other_imag - inp_imag * other_real

    # # Case 4: inp_is_conj = True, other_is_conj = True
    # out_real = inp_real * other_real + inp_imag * other_imag
    # out_imag = inp_real * other_imag - inp_imag * other_real
    if not inp_is_conj and not other_is_conj:  # Case 1
        out_real = tl.sum(inp_real * other_real + inp_imag * other_imag)
        out_imag = tl.sum(inp_real * other_imag - inp_imag * other_real)
    elif inp_is_conj and not other_is_conj:  # Case 2
        out_real = tl.sum(inp_real * other_real - inp_imag * other_imag)
        out_imag = tl.sum(inp_real * other_imag + inp_imag * other_real)
    elif not inp_is_conj and other_is_conj:  # Case 3
        out_real = tl.sum(inp_real * other_real - inp_imag * other_imag)
        out_imag = tl.sum(-inp_real * other_imag - inp_imag * other_real)
    else:  # Case 4
        out_real = tl.sum(inp_real * other_real + inp_imag * other_imag)
        out_imag = tl.sum(-inp_real * other_imag + inp_imag * other_real)

    return out_real, out_imag


# support old version triton which do not support tl.split
@libentry()
@triton.jit()
def vdot_kernel_complex(
    inp_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    inp_is_conj: tl.constexpr,
    other_is_conj: tl.constexpr,
    inp_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_stride = num_progs * BLOCK_SIZE

    acc_real = tl.zeros([], dtype=tl.float32)
    acc_imag = tl.zeros([], dtype=tl.float32)

    for current_start in range(0, n_elements // 2, grid_stride):
        complex_idx = current_start + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = complex_idx < n_elements // 2

        real_offset = complex_idx * 2

        inp_real = tl.load(inp_ptr + real_offset * inp_stride, mask=mask, other=0.0)
        inp_imag = tl.load(inp_ptr + real_offset * inp_stride + 1, mask=mask, other=0.0)

        other_real = tl.load(
            other_ptr + real_offset * other_stride, mask=mask, other=0.0
        )
        other_imag = tl.load(
            other_ptr + real_offset * other_stride + 1, mask=mask, other=0.0
        )

        out_real, out_imag = compute_vdot(
            inp_real, inp_imag, other_real, other_imag, inp_is_conj, other_is_conj
        )
        acc_real += out_real
        acc_imag += out_imag

    temp_offset = pid * 2
    tl.store(out_ptr + temp_offset, acc_real)
    tl.store(out_ptr + temp_offset + 1, acc_imag)


@libentry()
@triton.jit()
def reduce_kernel_complex(input_ptr, out_ptr, n_blocks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base_offset = tl.arange(0, BLOCK_SIZE)
    mask = base_offset < n_blocks

    inp_real = tl.load(input_ptr + base_offset * 2, mask=mask, other=0.0)
    inp_imag = tl.load(input_ptr + base_offset * 2 + 1, mask=mask, other=0.0)
    final_out_real = tl.sum(inp_real)
    final_out_imag = tl.sum(inp_imag)
    if pid == 0:
        tl.store(out_ptr, final_out_real)
        tl.store(out_ptr + 1, final_out_imag)


# only support real number
@libentry()
@triton.heuristics(runtime.get_heuristic_config("vdot"))
@triton.jit()
def dot_kernel(
    inp_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    inp_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    grid_stride = num_progs * BLOCK_SIZE

    acc = tl.zeros([], dtype=tl.float32)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    for current_start in range(0, n_elements, grid_stride):
        cur_offsets = current_start + offsets
        mask = cur_offsets < n_elements

        inp = tl.load(inp_ptr + inp_stride * cur_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        other = tl.load(
            other_ptr + other_stride * cur_offsets, mask=mask, other=0.0
        ).to(tl.float32)

        acc += tl.sum(inp * other)

    tl.store(out_ptr + pid, acc)


@libentry()
@triton.jit()
def reduce_kernel(
    partial_sums_ptr,
    output_ptr,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < n_blocks

    partial_sums = tl.load(partial_sums_ptr + offset, mask=mask, other=0.0)
    final_sum = tl.sum(partial_sums)

    if tl.program_id(0) == 0:
        tl.store(output_ptr, final_sum)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("vdot"))
@triton.jit()
def dot_kernel_fp32(
    inp_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    inp_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp = tl.load(inp_ptr + inp_stride * offset, mask=mask)
    other = tl.load(other_ptr + other_stride * offset, mask=mask)

    out = tl.sum(inp * other)
    tl.atomic_add(out_ptr, out)


def vdot(input: Tensor, other: Tensor):
    logger.debug("GEMS VDOT")

    assert (
        input.dtype == other.dtype
    ), f"Input tensors must have the same dtype. Got {input.dtype} and {other.dtype}."
    assert (
        input.ndim == 1 and other.ndim == 1
    ), f"Input tensors must be 1D. Got {input.ndim}D and {other.ndim}D."
    assert (
        input.size() == other.size()
    ), f"Input tensors must have the same size. Got {input.size()} and {other.size()}."

    inp = input
    inp_stride = inp.stride()[0]
    other_stride = other.stride()[0]

    if inp.is_complex():
        inp_is_conj = False
        other_is_conj = False

        if inp.is_conj():
            inp_is_conj = True
            inp = inp.conj()

        if other.is_conj():
            other_is_conj = True
            other = other.conj()

        inp_real = torch.view_as_real(inp)
        other_real = torch.view_as_real(other)

        n_elements = inp_real.numel()
        n_complex = inp.numel()

        block_size = runtime.get_heuristic_config("vdot")["BLOCK_SIZE"](
            {"n_elements": n_elements}
        )
        num_blocks = triton.cdiv(n_complex, block_size)

        grid_size = min(num_blocks, 1024)

        partial_real_sums = torch.empty(
            grid_size, dtype=inp_real.dtype, device=inp.device
        )
        grid = (grid_size,)
        vdot_kernel_complex[grid](
            inp_real,
            other_real,
            partial_real_sums,
            n_elements=n_elements,
            inp_is_conj=inp_is_conj,
            other_is_conj=other_is_conj,
            inp_stride=inp_stride,
            other_stride=other_stride,
            BLOCK_SIZE=block_size,
        )
        output_real = torch.empty(2, dtype=inp_real.dtype, device=inp.device)
        reduce_kernel_complex[(1,)](
            partial_real_sums,
            output_real,
            grid_size,
            BLOCK_SIZE=triton.next_power_of_2(grid_size),
        )
        return torch.view_as_complex(output_real)
    elif inp.dtype == torch.float32:
        output = torch.zeros([], dtype=torch.float32, device=inp.device)
        n_elements = inp.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        dot_kernel_fp32[grid](
            inp,
            other,
            output,
            n_elements=n_elements,
            inp_stride=inp_stride,
            other_stride=other_stride,
        )
        return output
    else:
        n_elements = inp.numel()
        block_size = runtime.get_heuristic_config("vdot")["BLOCK_SIZE"](
            {"n_elements": n_elements}
        )

        num_blocks = triton.cdiv(n_elements, block_size)
        grid_size = min(num_blocks, 1024)

        grid = (num_blocks,)
        partial_sums = torch.empty(grid_size, dtype=torch.float32, device=inp.device)
        dot_kernel[grid](
            inp,
            other,
            partial_sums,
            n_elements=n_elements,
            inp_stride=inp_stride,
            other_stride=other_stride,
            BLOCK_SIZE=block_size,
        )
        output = torch.empty([], dtype=input.dtype, device=inp.device)
        reduce_bs = min(triton.next_power_of_2(grid_size), 1024)
        reduce_kernel[(1,)](
            partial_sums,
            output,
            num_blocks,
            BLOCK_SIZE=reduce_bs,
        )
        return output
