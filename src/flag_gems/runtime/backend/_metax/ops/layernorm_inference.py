# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Sequence, Union

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as gl
except ImportError:
    HAS_GLUON = False
else:
    HAS_GLUON = True


if HAS_GLUON:

    @gluon.jit
    def layernorm_split_kernel(
        Input,
        Weight,
        Bias,
        Output,
        M: gl.constexpr,
        N: gl.constexpr,
        EPS: gl.constexpr,
        BODY: gl.constexpr,
        TAIL: gl.constexpr,
        VECTOR: gl.constexpr,
        HAS_WEIGHT: gl.constexpr = True,
        HAS_BIAS: gl.constexpr = True,
    ):
        body_layout: gl.constexpr = gl.BlockedLayout(
            [1, VECTOR], [64, 1], [1, 1], [1, 0]
        )
        tail_vector: gl.constexpr = min(VECTOR, TAIL // 64)
        tail_layout: gl.constexpr = gl.BlockedLayout(
            [1, tail_vector], [64, 1], [1, 1], [1, 0]
        )
        lane_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        lanes = gl.arange(0, 64, layout=gl.SliceLayout(1, body_layout))
        body_elements = gl.arange(0, BODY // 64, layout=gl.SliceLayout(0, body_layout))
        body_cols = (
            body_elements[None, :] // VECTOR * 64 + lanes[:, None]
        ) * VECTOR + body_elements[None, :] % VECTOR
        tail_lanes = gl.arange(0, 64, layout=gl.SliceLayout(1, tail_layout))
        tail_elements = gl.arange(0, TAIL // 64, layout=gl.SliceLayout(0, tail_layout))
        tail_cols = (
            BODY
            + (tail_elements[None, :] // tail_vector * 64 + tail_lanes[:, None])
            * tail_vector
            + tail_elements[None, :] % tail_vector
        )
        row = gl.program_id(0)
        body = gl.load(Input + row * N + body_cols).to(gl.float32)
        tail = gl.load(Input + row * N + tail_cols).to(gl.float32)
        body_sum = gl.convert_layout(gl.sum(body, 1), lane_layout)
        tail_sum = gl.convert_layout(gl.sum(tail, 1), lane_layout)
        mean = gl.sum(body_sum + tail_sum, 0) / N
        body_delta = body - mean
        tail_delta = tail - mean
        body_squares = gl.convert_layout(
            gl.sum(body_delta * body_delta, 1), lane_layout
        )
        tail_squares = gl.convert_layout(
            gl.sum(tail_delta * tail_delta, 1), lane_layout
        )
        variance = gl.sum(body_squares + tail_squares, 0) / N
        rstd = gl.rsqrt(variance + EPS)
        body_out = body_delta * rstd
        tail_out = tail_delta * rstd
        if HAS_WEIGHT:
            body_out = body_out * gl.load(Weight + body_cols).to(gl.float32)
            tail_out = tail_out * gl.load(Weight + tail_cols).to(gl.float32)
        if HAS_BIAS:
            body_out = body_out + gl.load(Bias + body_cols).to(gl.float32)
            tail_out = tail_out + gl.load(Bias + tail_cols).to(gl.float32)
        gl.store(Output + row * N + body_cols, body_out)
        gl.store(Output + row * N + tail_cols, tail_out)

    @gluon.jit
    def layernorm_vector_kernel(
        Input,
        Weight,
        Bias,
        Output,
        M: gl.constexpr,
        N: gl.constexpr,
        EPS: gl.constexpr,
        HAS_WEIGHT: gl.constexpr,
        HAS_BIAS: gl.constexpr,
        ROWS: gl.constexpr,
        COLS: gl.constexpr,
        WARPS: gl.constexpr,
        VECTOR: gl.constexpr,
    ):
        # One 64-lane warp owns each row; reduction never crosses warp boundaries.
        layout: gl.constexpr = gl.BlockedLayout(
            [1, VECTOR], [1, 64], [WARPS, 1], [1, 0]
        )
        rows = gl.program_id(0) * ROWS + gl.arange(
            0, ROWS, layout=gl.SliceLayout(1, layout)
        )
        cols = gl.arange(0, COLS, layout=gl.SliceLayout(0, layout))
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        values = gl.load(Input + rows[:, None] * N + cols[None, :], mask, other=0).to(
            gl.float32
        )
        mean = gl.sum(values, 1) / N
        centered = values - mean[:, None]
        centered_masked = gl.where(cols[None, :] < N, centered, 0.0)
        variance = gl.sum(centered_masked * centered_masked, 1) / N
        normalized = centered * gl.rsqrt(variance[:, None] + EPS)
        if HAS_WEIGHT:
            weight = gl.load(Weight + cols, cols < N, other=0).to(gl.float32)
            normalized = normalized * weight[None, :]
        if HAS_BIAS:
            bias = gl.load(Bias + cols, cols < N, other=0).to(gl.float32)
            normalized = normalized + bias[None, :]
        gl.store(Output + rows[:, None] * N + cols[None, :], normalized, mask)


@triton.jit
def layernorm_kernel(
    Input,
    Weight,
    Bias,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    NORMALIZED: tl.constexpr,
    WSTRIDES: tl.constexpr,
    BSTRIDES: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    if CONTIGUOUS:
        offsets = rows[:, None] * N + cols[None, :]
    else:
        row_offsets = tl.full((ROWS,), 0, tl.int64)
        col_offsets = tl.full((COLS,), 0, tl.int64)
        row_index = rows
        col_index = cols
        for dim in tl.static_range(len(SHAPE) - len(NORMALIZED) - 1, -1, -1):
            row_offsets += (row_index % SHAPE[dim]) * STRIDES[dim]
            row_index = row_index // SHAPE[dim]
        for dim in tl.static_range(len(NORMALIZED) - 1, -1, -1):
            col_offsets += (col_index % NORMALIZED[dim]) * STRIDES[
                len(SHAPE) - len(NORMALIZED) + dim
            ]
            col_index = col_index // NORMALIZED[dim]
        offsets = row_offsets[:, None] + col_offsets[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    values = tl.load(Input + offsets, mask, other=0).to(tl.float32)
    mean = tl.sum(values, 1) / N
    centered = values - mean[:, None]
    centered_masked = tl.where(cols[None, :] < N, centered, 0.0)
    variance = tl.sum(centered_masked * centered_masked, 1) / N
    normalized = centered * tl.rsqrt(variance[:, None] + EPS)
    if HAS_WEIGHT:
        weight_offsets = tl.full((COLS,), 0, tl.int64)
        index = cols
        for dim in tl.static_range(len(NORMALIZED) - 1, -1, -1):
            weight_offsets += (index % NORMALIZED[dim]) * WSTRIDES[dim]
            index = index // NORMALIZED[dim]
        weight = tl.load(Weight + weight_offsets, cols < N, other=0).to(tl.float32)
        normalized = normalized * weight[None, :]
    if HAS_BIAS:
        bias_offsets = tl.full((COLS,), 0, tl.int64)
        index = cols
        for dim in tl.static_range(len(NORMALIZED) - 1, -1, -1):
            bias_offsets += (index % NORMALIZED[dim]) * BSTRIDES[dim]
            index = index // NORMALIZED[dim]
        bias = tl.load(Bias + bias_offsets, cols < N, other=0).to(tl.float32)
        normalized = normalized + bias[None, :]
    tl.store(Output + rows[:, None] * N + cols[None, :], normalized, mask)


def layernorm(
    input: torch.Tensor,
    normalized_shape: Union[int, Sequence[int]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """C550 inference LayerNorm with FP32 centered variance accumulation."""
    normalized_shape = (
        (normalized_shape,)
        if isinstance(normalized_shape, int)
        else tuple(normalized_shape)
    )
    if not normalized_shape or any(size <= 0 for size in normalized_shape):
        raise ValueError("normalized_shape must contain positive dimensions")
    if tuple(input.shape[-len(normalized_shape) :]) != normalized_shape:
        raise ValueError("normalized_shape must match the trailing input dimensions")
    if input.device.type != "cuda":
        raise NotImplementedError("layernorm requires a MetaX CUDA-compatible device")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("layernorm supports FP16, BF16 and FP32")
    width = math.prod(normalized_shape)
    if width > 65536:
        raise NotImplementedError("normalized size above 65536 is unsupported")
    for affine in (weight, bias):
        if affine is None:
            continue
        if tuple(affine.shape) != normalized_shape:
            raise ValueError("affine shape must equal normalized_shape")
        if affine.device != input.device or affine.dtype != input.dtype:
            raise ValueError("affine device and dtype must match input")
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in (input, weight, bias)
    ):
        raise NotImplementedError("layernorm supports inference without autograd")
    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    rows = input.numel() // width
    if rows == 0:
        return output
    # C550 measurements favor a single 64-lane warp over cross-warp reductions.
    tile_rows = 1
    warps = 1 if width <= 4096 else (4 if width <= 8192 else 8)
    is_dense_affine = all(
        affine is None or affine.is_contiguous() for affine in (weight, bias)
    )
    if HAS_GLUON and input.is_contiguous() and is_dense_affine and width <= 4096:
        body = 1 << (width.bit_length() - 1)
        tail = width - body
        # Exact power-of-two segments avoid masked padding and its live registers.
        use_split = (
            rows >= 128 and body >= 256 and tail >= 64 and tail & (tail - 1) == 0
        )
        with torch.cuda.device(input.device):
            if use_split:
                layernorm_split_kernel[(rows,)](
                    input,
                    weight,
                    bias,
                    output,
                    rows,
                    width,
                    eps,
                    body,
                    tail,
                    4,
                    weight is not None,
                    bias is not None,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
            else:
                layernorm_vector_kernel[(rows,)](
                    input,
                    weight,
                    bias,
                    output,
                    rows,
                    width,
                    eps,
                    weight is not None,
                    bias is not None,
                    1,
                    triton.next_power_of_2(width),
                    1,
                    4,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
        return output
    with torch.cuda.device(input.device):
        layernorm_kernel[(triton.cdiv(rows, tile_rows),)](
            input,
            weight,
            bias,
            output,
            rows,
            width,
            eps,
            tuple(input.shape),
            tuple(input.stride()),
            normalized_shape,
            tuple(weight.stride()) if weight is not None else (),
            tuple(bias.stride()) if bias is not None else (),
            input.is_contiguous(),
            weight is not None,
            bias is not None,
            tile_rows,
            triton.next_power_of_2(width),
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return output
