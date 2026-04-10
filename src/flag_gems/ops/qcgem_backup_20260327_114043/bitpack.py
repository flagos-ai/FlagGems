# SPDX-License-Identifier: Apache-2.0
# QC-GEM: Quantized Computing GEM library for FlagGems

import torch
import triton
import triton.language as tl

from .dtypes import TORCH_DTYPE_TO_TRITON, PACKING_BITWIDTH_TO_TORCH_DTYPE


_powers_of_2 = [2**n for n in range(10)][::-1]


def highest_divisor(n: int, max_val: int) -> int:
    if max_val == 1:
        return 1
    for d in _powers_of_2:
        if n % d == 0 and d <= max_val:
            return d
    return 1


@triton.jit
def or_fn(a, b):
    return a | b


@triton.jit
def pack_weights_over_cols_kernel(
    W_q_ptr,
    W_q_out_ptr,
    num_input_cols,
    num_cols,
    unroll: tl.constexpr,
    elements_per_sample: tl.constexpr,
    W_nbits: tl.constexpr,
    out_dtype: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_row = (pid // num_cols) * unroll
    pid_col = (pid % num_cols)

    for r in range(unroll):
        start_col = pid_col * elements_per_sample
        cols = tl.arange(0, elements_per_sample)
        shifts = (cols * W_nbits).to(out_dtype)

        offset = pid_row * num_input_cols + start_col + cols
        offset = tl.max_contiguous(tl.multiple_of(offset, elements_per_sample), elements_per_sample)
        values = tl.load(W_q_ptr + offset).to(out_dtype)

        result = tl.reduce(values << shifts, axis=0, combine_fn=or_fn)

        output_offset = pid_row * num_cols + pid_col
        tl.store(W_q_out_ptr + output_offset, result)
        pid_row += 1


def pack_weights_over_cols_triton(W_q, W_nbits, packing_bitwidth, transpose):
    assert packing_bitwidth in [8, 16, 32, 64], "Unsupported bitpacking width"
    assert W_nbits in [8, 4, 2, 1], "Unsupported nbits"
    elements_per_sample = packing_bitwidth // W_nbits
    num_rows, num_input_cols = W_q.shape
    num_cols = num_input_cols // elements_per_sample

    dtype = PACKING_BITWIDTH_TO_TORCH_DTYPE[packing_bitwidth]
    out_dtype = TORCH_DTYPE_TO_TRITON[dtype]

    W_q_out = torch.empty((num_rows, num_cols), dtype=dtype, device=W_q.device)
    unroll = highest_divisor(num_rows, max_val=64)
    grid = (triton.cdiv(num_rows * num_cols, unroll),)

    pack_weights_over_cols_kernel[grid](
        W_q.contiguous(),
        W_q_out,
        num_input_cols,
        num_cols,
        unroll,
        elements_per_sample,
        W_nbits,
        out_dtype,
        num_stages=2,
        num_warps=1,
    )

    if transpose:
        W_q_out = W_q_out.t()

    return W_q_out, elements_per_sample


def pack_weights_over_cols_torch(W_q, W_nbits, packing_bitwidth, transpose):
    assert packing_bitwidth in [8, 16, 32, 64], "Unsupported bitpacking width"
    assert W_nbits in [8, 4, 2, 1], "Unsupported nbits"
    elements_per_sample = packing_bitwidth // W_nbits

    W_q = W_q.to(torch.int32)
    W_q_out = torch.zeros(
        (W_q.shape[0], W_q.shape[1] // elements_per_sample),
        dtype=torch.int32 if packing_bitwidth <= 32 else torch.int64,
        device=W_q.device,
    )

    for j in range(W_q.shape[1]):
        col = j // elements_per_sample
        shift = (j % elements_per_sample) * W_nbits
        W_q_out[:, col] |= W_q[:, j] << shift

    W_q_out = W_q_out.to(dtype=PACKING_BITWIDTH_TO_TORCH_DTYPE[packing_bitwidth])

    if transpose:
        W_q_out = W_q_out.t()

    return W_q_out, elements_per_sample


pack_weights_over_cols = pack_weights_over_cols_triton
