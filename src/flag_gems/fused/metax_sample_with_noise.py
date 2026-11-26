# SPDX-License-Identifier: Apache-2.0
"""Two-stage exponential-race selection for MetaX, using vLLM's FP32 probabilities and noise."""

import torch
import triton
import triton.language as tl

import flag_gems


@triton.jit
def _race_tiles(
    Probs,
    Noise,
    Values,
    Indices,
    N: tl.constexpr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
    GEMS_ARGMAX: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    prob = tl.load(Probs + row * N + col, valid, other=0.0)
    noise = tl.load(Noise + row * N + col, valid, other=1.0)
    score = prob / noise
    if GEMS_ARGMAX:
        # Match the locked FlagGems long-row reduction, including a tile with NaNs.
        value, index = tl.max(
            tl.where(valid, score, -3.4028234663852886e38),
            0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        tl.store(Values + row * NT + tile, value)
        tl.store(Indices + row * NT + tile, tile * BLOCK + index)
        return
    nan_col = tl.min(tl.where(valid & (score != score), col, N), 0)
    maximum = tl.max(tl.where(valid & (score == score), score, -float("inf")), 0)
    winner = tl.min(tl.where(valid & (score == maximum), col, N), 0)
    # torch.argmax uses the first index, including for NaNs.
    tl.store(Values + row * NT + tile, tl.where(nan_col < N, float("nan"), maximum))
    tl.store(Indices + row * NT + tile, tl.where(nan_col < N, nan_col, winner))


@triton.jit
def _race_winner(
    Values,
    Indices,
    Out,
    N: tl.constexpr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
    GEMS_ARGMAX: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    value = tl.load(Values + row * NT + tile, tile < NT, other=-float("inf"))
    index = tl.load(Indices + row * NT + tile, tile < NT, other=N)
    nan_index = tl.min(tl.where(value != value, index, N), 0)
    maximum = tl.max(tl.where(value == value, value, -float("inf")), 0)
    winner = tl.min(tl.where(value == maximum, index, N), 0)
    if GEMS_ARGMAX:
        result = tl.where(maximum > -3.4028234663852886e38, winner, 0)
    else:
        result = tl.where(nan_index < N, nan_index, winner)
    tl.store(Out + row, result.to(tl.int64))


def metax_sample_with_noise(probs, noise):
    rows, width = probs.shape
    registrar = getattr(flag_gems, "current_work_registrar", None)
    gems_argmax = registrar is not None and "argmax" in registrar.get_all_keys()
    block = 4096 if gems_argmax else 2048
    tiles = triton.cdiv(width, block)
    values = torch.empty((rows, tiles), dtype=torch.float32, device=probs.device)
    indices = torch.empty((rows, tiles), dtype=torch.int32, device=probs.device)
    result = torch.empty(rows, dtype=torch.int64, device=probs.device)
    with torch.cuda.device(probs.device):
        _race_tiles[(rows, tiles)](
            probs, noise, values, indices, width, tiles, block, gems_argmax, num_warps=8
        )
        _race_winner[(rows,)](
            values,
            indices,
            result,
            width,
            tiles,
            triton.next_power_of_2(tiles),
            gems_argmax,
            num_warps=1,
        )
    return result
