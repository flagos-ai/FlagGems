# SPDX-License-Identifier: Apache-2.0
"""Sample from already filtered logits using per-tile softmax mass."""

import torch
import triton
import triton.language as tl


@triton.jit
def _tile_mass(X, M, S, stride, V: tl.constexpr, T: tl.constexpr, WIDTH: tl.constexpr):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    cols = tile * WIDTH + tl.arange(0, WIDTH)
    x = tl.load(X + row.to(tl.int64) * stride + cols, cols < V, -float("inf"))
    maximum = tl.max(x, 0)
    invalid = tl.sum(((x != x) | (x == float("inf"))).to(tl.int32), 0) != 0
    finite_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    mass = tl.sum(tl.exp(x - finite_max), 0)
    tl.store(M + row * T + tile, tl.where(invalid, float("inf"), maximum))
    tl.store(S + row * T + tile, mass)


@triton.jit
def _draw(
    X,
    M,
    S,
    U,
    OUT,
    stride,
    V: tl.constexpr,
    T: tl.constexpr,
    TILE_LANES: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    tiles = tl.arange(0, TILE_LANES)
    maxima = tl.load(M + row * T + tiles, tiles < T, -float("inf"))
    masses = tl.load(S + row * T + tiles, tiles < T, 0.0)
    maximum = tl.max(maxima, 0)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = masses * tl.exp(maxima - safe_max)
    prefix = tl.cumsum(weights, 0)
    total = tl.max(prefix, 0)
    target = tl.load(U + row) * total
    # A rounded endpoint belongs to the final positive-mass tile.
    last_tile = tl.max(tl.where(weights > 0, tiles, 0), 0)
    first = tl.min(tl.where((weights > 0) & (prefix > target), tiles, T), 0)
    tile = tl.where(first < T, first, last_tile)
    before = tl.max(tl.where(tiles < tile, prefix, 0.0), 0)
    tile_weight = tl.sum(tl.where(tiles == tile, weights, 0.0), 0)
    within = (target - before) / tl.where(tile_weight > 0, tile_weight, 1.0)

    cols = tile * WIDTH + tl.arange(0, WIDTH)
    x = tl.load(X + row.to(tl.int64) * stride + cols, cols < V, -float("inf"))
    local_max = tl.max(x, 0)
    w = tl.exp(x - tl.where(local_max == -float("inf"), 0.0, local_max))
    cum = tl.cumsum(w, 0)
    local_target = tl.maximum(within, 0.0) * tl.max(cum, 0)
    first_col = tl.min(tl.where((w > 0) & (cum > local_target), cols, V), 0)
    last_col = tl.max(tl.where(w > 0, cols, 0), 0)
    chosen = tl.where(first_col < V, first_col, last_col)
    tl.store(OUT + row, tl.where((total > 0) & (maximum < float("inf")), chosen, 0))


def draw_filtered_logits(logits, uniforms=None):
    """Return int64 token IDs for rows with contiguous float32 vocabularies.

    Normalization uses every unmasked token. Sampling scratch scales with
    the number of vocabulary tiles. All tensors are private to the call and
    CUDA graph capture owns their addresses across replays.
    """
    rows, vocab = logits.shape
    width = 1024
    tiles = triton.cdiv(vocab, width)
    if uniforms is None:
        uniforms = torch.rand(rows, device=logits.device, dtype=torch.float32)
    with torch.cuda.device(logits.device):
        workspace = logits.new_empty((2, rows, tiles))
        maxima, masses = workspace.unbind(0)
        out = torch.empty(rows, device=logits.device, dtype=torch.int64)
        _tile_mass[(tiles, rows)](
            logits, maxima, masses, logits.stride(0), vocab, tiles, width, num_warps=4
        )
        _draw[(rows,)](
            logits,
            maxima,
            masses,
            uniforms,
            out,
            logits.stride(0),
            vocab,
            tiles,
            triton.next_power_of_2(tiles),
            width,
            num_warps=4,
        )
    return out
