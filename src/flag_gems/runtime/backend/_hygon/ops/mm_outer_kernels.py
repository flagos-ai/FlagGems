# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""K=1 product tiles that reuse the input row and column vectors."""

import triton
import triton.language as tl


@triton.jit
def outer_product(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    AM: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    STORE: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    a = tl.load(A + rows * AM, rows < M, 0)
    b = tl.load(B + cols * BN, cols < N, 0)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        a[:, None] * b[None, :],
        (rows[:, None] < M) & (cols[None, :] < N),
        cache_modifier=STORE,
    )
