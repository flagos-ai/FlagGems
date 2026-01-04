import math

import torch
import torch.profiler
import triton
import triton.language as tl


@triton.jit
def bit_reverse_kernel(real_in, imag_in, real_out, imag_out, n):
    """do reverse first: input[i] -> output[bit_reverse(i)]"""
    tid = tl.program_id(0)
    if tid >= n:
        return

    # compute bits & reverse
    temp_n = n
    idx = tid
    rev_idx = 0
    temp_idx = idx
    while temp_n > 1:
        temp_n //= 2
        rev_idx = (rev_idx << 1) | (temp_idx & 1)
        temp_idx = temp_idx >> 1

    val_real = tl.load(real_in + idx)
    val_imag = tl.load(imag_in + idx)
    tl.store(real_out + rev_idx, val_real)
    tl.store(imag_out + rev_idx, val_imag)


@triton.jit
def fft_stage_kernel(real_ptr, imag_ptr, n, stage):
    """iterate the FFT stage"""
    PI = math.pi
    tid = tl.program_id(0)

    if tid >= n // 2:
        return

    # compute current parameter
    half_block = 1 << (stage - 1)  # 2^stage

    # Each thread processes one butterfly pair
    butterfly_group = tid // half_block
    pos_in_group = tid % half_block

    # compute the index of two elements in butterfly pair
    first_idx = butterfly_group * half_block * 2 + pos_in_group
    second_idx = first_idx + half_block

    if second_idx >= n:
        return

    # load
    a_real = tl.load(real_ptr + first_idx)
    a_imag = tl.load(imag_ptr + first_idx)
    b_real = tl.load(real_ptr + second_idx)
    b_imag = tl.load(imag_ptr + second_idx)

    # calculate complex amplitude
    angle = PI * pos_in_group / half_block
    w_real = tl.cos(-angle)
    w_imag = tl.sin(-angle)

    tw_real = b_real * w_real - b_imag * w_imag
    tw_imag = b_real * w_imag + b_imag * w_real

    # butterfly
    result_a_real = a_real + tw_real
    result_a_imag = a_imag + tw_imag
    result_b_real = a_real - tw_real
    result_b_imag = a_imag - tw_imag

    # store
    tl.store(real_ptr + first_idx, result_a_real)
    tl.store(imag_ptr + first_idx, result_a_imag)
    tl.store(real_ptr + second_idx, result_b_real)
    tl.store(imag_ptr + second_idx, result_b_imag)


def fft_1d(x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    N = x.shape[0]
    # make sure N is an integer power of 2
    assert N > 0 and (N & (N - 1)) == 0

    x_real = x.real.clone()
    x_imag = x.imag.clone()

    temp_real = torch.zeros_like(x_real)
    temp_imag = torch.zeros_like(x_imag)
    bit_reverse_kernel[(N,)](x_real, x_imag, temp_real, temp_imag, N)

    x_real.copy_(temp_real)
    x_imag.copy_(temp_imag)

    log2n = N.bit_length() - 1
    for stage in range(1, log2n + 1):
        fft_stage_kernel[(N // 2,)](x_real, x_imag, N, stage)

    output.real.copy_(x_real)
    output.imag.copy_(x_imag)
    return output
