# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl


@triton.jit
def _merge_moments(count, mean, square, add_count, add_mean, add_square):
    total = count + add_count
    ratio = tl.div_rn(add_count.to(tl.float32), total.to(tl.float32))
    ratio = tl.where(total == 0, 0.0, ratio)
    delta = add_mean - mean
    scaled_delta = ratio * delta
    merged_mean = mean + scaled_delta
    merged_square = tl.fma(
        delta * count.to(tl.float32), scaled_delta, square + add_square
    )
    return total, merged_mean, merged_square


@triton.jit
def rowwise_cascade_moments(
    in_ptr, rows, row_mask, N: tl.constexpr, DEPTH: tl.constexpr, STACK: tl.constexpr
):
    # Chunked Welford moments with a fixed cascade and lane-fold order.
    # Explicit FMA keeps updates stable for ill-conditioned input rows.
    R: tl.constexpr = rows.shape[0]
    lanes = tl.arange(0, 8)
    levels = tl.arange(0, STACK)
    counts = tl.zeros((STACK,), tl.int32)
    means = tl.zeros((R, STACK, 8), tl.float32)
    squares = tl.zeros((R, STACK, 8), tl.float32)
    vectors: tl.constexpr = N // 8
    for chunk in range(tl.cdiv(vectors, 16)):
        chunk_count = tl.minimum(16, vectors - chunk * 16)
        cm = tl.zeros((R, 8), tl.float32)
        cs = tl.zeros((R, 8), tl.float32)
        for j in range(chunk_count):
            offsets = (chunk * 16 + j) * 8 + lanes
            x = tl.load(
                in_ptr + rows[:, None] * N + offsets[None, :],
                row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            delta = x - cm
            cm = tl.fma(tl.div_rn(1.0, (j + 1).to(tl.float32)), delta, cm)
            cs = tl.fma(delta, x - cm, cs)
        c0 = tl.sum(tl.where(levels == 0, counts, 0), 0)
        m0 = tl.sum(tl.where((levels == 0)[None, :, None], means, 0.0), 1)
        s0 = tl.sum(tl.where((levels == 0)[None, :, None], squares, 0.0), 1)
        c0, m0, s0 = _merge_moments(c0, m0, s0, chunk_count, cm, cs)
        counts = tl.where(levels == 0, c0, counts)
        means = tl.where((levels == 0)[None, :, None], m0[:, None, :], means)
        squares = tl.where((levels == 0)[None, :, None], s0[:, None, :], squares)
        for level in tl.static_range(1, DEPTH):
            if ((chunk + 1) & ((1 << level) - 1)) == 0:
                lower_c = tl.sum(tl.where(levels == level - 1, counts, 0), 0)
                lower_m = tl.sum(
                    tl.where((levels == level - 1)[None, :, None], means, 0.0), 1
                )
                lower_s = tl.sum(
                    tl.where((levels == level - 1)[None, :, None], squares, 0.0), 1
                )
                upper_c = tl.sum(tl.where(levels == level, counts, 0), 0)
                upper_m = tl.sum(
                    tl.where((levels == level)[None, :, None], means, 0.0), 1
                )
                upper_s = tl.sum(
                    tl.where((levels == level)[None, :, None], squares, 0.0), 1
                )
                merged_c, merged_m, merged_s = _merge_moments(
                    upper_c, upper_m, upper_s, lower_c, lower_m, lower_s
                )
                counts = tl.where(
                    levels == level, merged_c, tl.where(levels == level - 1, 0, counts)
                )
                means = tl.where(
                    (levels == level)[None, :, None],
                    merged_m[:, None, :],
                    tl.where((levels == level - 1)[None, :, None], 0.0, means),
                )
                squares = tl.where(
                    (levels == level)[None, :, None],
                    merged_s[:, None, :],
                    tl.where((levels == level - 1)[None, :, None], 0.0, squares),
                )
    c0 = tl.sum(tl.where(levels == 0, counts, 0), 0)
    m0 = tl.sum(tl.where((levels == 0)[None, :, None], means, 0.0), 1)
    s0 = tl.sum(tl.where((levels == 0)[None, :, None], squares, 0.0), 1)
    for level in tl.static_range(1, DEPTH):
        add_c = tl.sum(tl.where(levels == level, counts, 0), 0)
        add_m = tl.sum(tl.where((levels == level)[None, :, None], means, 0.0), 1)
        add_s = tl.sum(tl.where((levels == level)[None, :, None], squares, 0.0), 1)
        c0, m0, s0 = _merge_moments(c0, m0, s0, add_c, add_m, add_s)
    # Scalar tail precedes the sequential fold of the eight vector lanes.
    count = tl.full((), 0, tl.int32)
    mean = tl.zeros((R,), tl.float32)
    square = tl.zeros((R,), tl.float32)
    for col in range(vectors * 8, N):
        x = tl.load(in_ptr + rows * N + col, row_mask, other=0.0).to(tl.float32)
        delta = x - mean
        count += 1
        mean += tl.div_rn(delta, count.to(tl.float32))
        square = tl.fma(delta, x - mean, square)
    for lane in tl.static_range(8):
        add_mean = tl.sum(tl.where((lanes == lane)[None, :], m0, 0.0), 1)
        add_square = tl.sum(tl.where((lanes == lane)[None, :], s0, 0.0), 1)
        total = count + vectors
        ratio = tl.div_rn(float(vectors), total.to(tl.float32))
        ratio = tl.where(total == 0, 0.0, ratio)
        delta = add_mean - mean
        mean = tl.fma(ratio, delta, mean)
        square += add_square + delta * delta * ratio * count.to(tl.float32)
        count = total
    return mean, tl.div_rn(square, float(N))
