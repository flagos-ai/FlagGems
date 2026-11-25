"""Fast Hadamard Transform in Triton (KunlunXin).

v0: Multi-pass butterfly via global memory, 1 kernel launch per stage.
Simple baseline for correctness. Each butterfly stage reads from IN, writes to OUT.
"""

import math

import torch
import triton
import triton.language as tl

from ..utils.tle_copy import tle_copy

MAX_GRID = 65535


@triton.jit
def _butterfly_stage(
    IN_ptr,
    OUT_ptr,
    stride_row,
    N_ROWS,
    ROWS_PER_PROGRAM: tl.constexpr,
    STRIDE_S: tl.constexpr,
    DIM: tl.constexpr,
):
    """One butterfly stage: read from IN, write to OUT.

    For each element at position i:
      partner = i ^ STRIDE_S
      if (i & STRIDE_S) == 0:  out[i] = in[i] + in[partner]
      else:                     out[i] = in[partner] - in[i]
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, DIM)

    for row_idx in tl.static_range(ROWS_PER_PROGRAM):
        row_id = pid * ROWS_PER_PROGRAM + row_idx
        if row_id < N_ROWS:
            base = row_id * stride_row

            x = tl.load(IN_ptr + base + offsets).to(tl.float32)
            partner_offsets = offsets ^ STRIDE_S
            x_partner = tl.load(IN_ptr + base + partner_offsets).to(tl.float32)

            is_upper = (offsets & STRIDE_S) == 0
            result = tl.where(is_upper, x + x_partner, x_partner - x)

            tl.store(OUT_ptr + base + offsets, result)


@triton.jit
def _scale_cast(
    IN_ptr,
    OUT_ptr,
    stride_in_row,
    stride_out_row,
    scale,
    N_ROWS,
    ROWS_PER_PROGRAM: tl.constexpr,
    DIM: tl.constexpr,
):
    """Scale fp32 buffer and cast to output dtype."""
    pid = tl.program_id(0)
    offsets = tl.arange(0, DIM)

    for row_idx in tl.static_range(ROWS_PER_PROGRAM):
        row_id = pid * ROWS_PER_PROGRAM + row_idx
        if row_id < N_ROWS:
            x = tl.load(IN_ptr + row_id * stride_in_row + offsets)
            tl.store(OUT_ptr + row_id * stride_out_row + offsets, x * scale)


def _hadamard_transform_fwd(x, scale):
    orig_shape = x.shape
    dim = x.shape[-1]
    input_dtype = x.dtype

    log_dim = math.ceil(math.log2(dim)) if dim > 0 else 0
    dim_padded = 1 << log_dim
    if dim != dim_padded:
        x_padded = torch.zeros(
            *x.shape[:-1], dim_padded, dtype=x.dtype, device=x.device
        )
        if not tle_copy(x, x_padded[..., :dim]):
            torch.ops.aten._copy_from(x, x_padded[..., :dim], False)
        x = x_padded

    x_flat = x.reshape(-1, dim_padded).contiguous()
    n_rows = x_flat.shape[0]
    n_stages = log_dim

    rows_per_prog = 1
    while (n_rows + rows_per_prog - 1) // rows_per_prog > MAX_GRID:
        rows_per_prog *= 2
    grid_size = (n_rows + rows_per_prog - 1) // rows_per_prog

    buf_a = x_flat.float().clone()
    buf_b = torch.empty_like(buf_a)

    stride_row = dim_padded

    for s in range(n_stages):
        stride_s = 1 << s
        _butterfly_stage[(grid_size,)](
            buf_a,
            buf_b,
            stride_row,
            n_rows,
            ROWS_PER_PROGRAM=rows_per_prog,
            STRIDE_S=stride_s,
            DIM=dim_padded,
        )
        buf_a, buf_b = buf_b, buf_a

    out = torch.empty(n_rows, dim_padded, dtype=input_dtype, device=x.device)
    _scale_cast[(grid_size,)](
        buf_a,
        out,
        stride_row,
        dim_padded,
        scale,
        n_rows,
        ROWS_PER_PROGRAM=rows_per_prog,
        DIM=dim_padded,
    )

    if dim != dim_padded:
        out = out[:, :dim]
    return out.reshape(orig_shape)


class HadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx._hadamard_transform_scale = scale
        return _hadamard_transform_fwd(x, scale)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _hadamard_transform_fwd(
                grad_output.contiguous(), ctx._hadamard_transform_scale
            ),
            None,
        )


def hadamard_transform(x, scale=1.0):
    """Fast Hadamard Transform.

    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix.
    Equivalent to F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale.
    If dim is not a power of 2, we implicitly pad x with zero so that dim is
    the next power of 2.
    """
    return HadamardTransformFn.apply(x, scale)


def hadamard_transform_12N(x, scale=1.0):
    """Hadamard transform for dim = 12 * 2^k (e.g. 12*512 = 6144)."""
    return HadamardTransformFn.apply(x, scale)


def hadamard_transform_20N(x, scale=1.0):
    """Hadamard transform for dim = 20 * 2^k (e.g. 20*1024 = 20480)."""
    return HadamardTransformFn.apply(x, scale)


def hadamard_transform_28N(x, scale=1.0):
    """Hadamard transform for dim = 28 * 2^k (e.g. 28*1024 = 28672)."""
    return HadamardTransformFn.apply(x, scale)


def hadamard_transform_40N(x, scale=1.0):
    """Hadamard transform for dim = 40 * 2^k (e.g. 40*1024 = 40960)."""
    return HadamardTransformFn.apply(x, scale)
