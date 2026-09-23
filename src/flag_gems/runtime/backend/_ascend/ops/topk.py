# Copyright 2026- Xcoresigma Technology Co., Ltd
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

"""
TopK on Ascend NPU via tle.dsa.ascend.raw
===================================================

This file keeps a generic fallback path and layers the current optimized
dispatch on top:

  - generic final: use round-robin sort plus generic merge+unpack for verified
    tiny-S, small-N 2048, direct final, and high-K medium-S cases.
  - round-robin sort path: sort proposal runs with length min(K, SEG_LEN).
  - tiny-S small-batch path: restore the older 2048-run + merge4 path for
    B<=4, S<=2, K<=512.
  - stream16 small-K merge path: route K <= 2048 through one cascade merge path.
  - batch-local path: merge per-row runs into out_segs, optionally group
    them further, then finish with merge+unpack.

This variant calls the unified TopK custom ops through tle.dsa.ascend.raw.
Data format: proposal = [value(f32), index(i32 as f32)] = 8B.
"""

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle as tle
    import triton.experimental.tle.language.dsa.ascend.custom_ops
    import triton.language.extra.cann.extension as al

    HAS_TLE = True
except (ImportError, AttributeError):
    tle = None
    al = None
    HAS_TLE = False

NUM_CORES = 48
CHUNK = 2048  # Max proposals copied per way in a single UB pass
# Chunk size for the intermediate per-core/group merges. Must be a power of
# two: both sides of a copy need equal static shapes (arange only takes powers
# of two), and 1024 keeps CHUNK_C*2 == npo2(CHUNK_C*2) so the existing subview
# pattern stays valid.
CORE_LOCAL_CHUNK = 1024
MAX_WAYS = 4
DIRECT_GROUP_MIN_SEGS = 16
DIRECT_GROUP_MAX_SEGS = 128
SMALLK_SORT_MAX_K = 512
LAYERED_SORT_MAX_K = 2048
TINY_S_SMALL_BATCH_MAX_K = 512
STREAM16_SMALLK_MERGE_MAX_K = 2048
STREAM16_SMALLK_GROUP_RUNS = 16
SMALLN_2048_MIN_N = 2049
SMALLN_2048_MAX_N = 8192


# ════════════════════════════════════════════════════════════════════════════
# Kernel 2-fused: generic merge tree + unpack.
#   Replaces the trailing separate merge and unpack launches.
#   The merge still reuses the WorkGM ping-pong regions; once done, the same
#   kernel reads the final region and unpacks it in chunks into Yv/Yi.
# ════════════════════════════════════════════════════════════════════════════
@triton.jit
def generic_merge_unpack_kernel(
    WorkGM,
    Yv,
    Yi,
    NUM_SEG: tl.constexpr,
    SEG_LEN: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    K: tl.constexpr,
    CHUNK_C: tl.constexpr,
    IN_CAP: tl.constexpr,
    OUT_CAP: tl.constexpr,
    MAX_STREAM_ITERS: tl.constexpr,
    MAX_WAYS_C: tl.constexpr,
    UNPACK_CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    base_p0 = (row * 2 + 0) * region_props
    base_p1 = (row * 2 + 1) * region_props

    cons = tl.zeros([4], dtype=tl.int32)
    CP2: tl.constexpr = triton.next_power_of_2(CHUNK_C * 2)
    OP2: tl.constexpr = triton.next_power_of_2(OUT_CAP)
    UP2: tl.constexpr = triton.next_power_of_2(UNPACK_CHUNK * 2)
    in_ub = tle.dsa.alloc([IN_CAP], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    out_ub = tle.dsa.alloc(
        [OUT_CAP + 8], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )

    cur_n = NUM_SEG
    cur_full = SEG_LEN
    cur_last = SEG_LEN
    src_phase = 0

    while cur_n > 1:
        stride = cur_full
        out_len = cur_full * MAX_WAYS_C
        if out_len > K:
            out_len = K
        out_n = (cur_n + MAX_WAYS_C - 1) // MAX_WAYS_C
        last_ways = cur_n - MAX_WAYS_C * (out_n - 1)
        last_total = (last_ways - 1) * cur_full + cur_last
        nxt_full = out_len
        nxt_last = last_total
        if nxt_last > K:
            nxt_last = K

        src_base = base_p0
        if src_phase == 1:
            src_base = base_p1
        dst_base = base_p1
        if src_phase == 1:
            dst_base = base_p0

        for g in tl.range(0, out_n):
            base_seg = g * MAX_WAYS_C
            ways = MAX_WAYS_C
            if base_seg + ways > cur_n:
                ways = cur_n - base_seg
            is_last_group = g == out_n - 1
            g0 = (base_seg + 0) * stride
            g1 = (base_seg + 1) * stride
            g2 = (base_seg + 2) * stride
            g3 = (base_seg + 3) * stride
            sl0 = cur_full
            sl1 = cur_full if ways > 1 else 0
            sl2 = cur_full if ways > 2 else 0
            sl3 = cur_full if ways > 3 else 0
            if is_last_group:
                if ways == 1:
                    sl0 = cur_last
                elif ways == 2:
                    sl1 = cur_last
                elif ways == 3:
                    sl2 = cur_last
                else:
                    sl3 = cur_last
            total = sl0 + sl1 + sl2 + sl3
            grp_cap = total
            if grp_cap > K:
                grp_cap = K
            dst_seg_off = g * out_len

            c0 = 0
            c1 = 0
            c2 = 0
            c3 = 0
            produced = 0
            it = 0
            while (produced < grp_cap) and (it < MAX_STREAM_ITERS):
                it += 1
                r0 = sl0 - c0
                r1 = sl1 - c1 if ways > 1 else 0
                r2 = sl2 - c2 if ways > 2 else 0
                r3 = sl3 - c3 if ways > 3 else 0
                l0 = r0 if r0 < CHUNK_C else CHUNK_C
                l1 = r1 if r1 < CHUNK_C else CHUNK_C
                l2 = r2 if r2 < CHUNK_C else CHUNK_C
                l3 = r3 if r3 < CHUNK_C else CHUNK_C
                rem_ways = 0
                if r0 > 0:
                    rem_ways += 1
                if r1 > 0:
                    rem_ways += 1
                if r2 > 0:
                    rem_ways += 1
                if r3 > 0:
                    rem_ways += 1
                aw = 0
                if l0 > 0:
                    aw += 1
                if l1 > 0:
                    aw += 1
                if l2 > 0:
                    aw += 1
                if l3 > 0:
                    aw += 1

                if rem_ways == 0:
                    produced = grp_cap
                elif rem_ways == 1:
                    soff = src_base + g0 + c0
                    sres = r0
                    if r1 > 0:
                        soff = src_base + g1 + c1
                        sres = r1
                    if r2 > 0:
                        soff = src_base + g2 + c2
                        sres = r2
                    if r3 > 0:
                        soff = src_base + g3 + c3
                        sres = r3
                    take = sres
                    if produced + take > grp_cap:
                        take = grp_cap - produced
                    cpos = 0
                    while cpos < take:
                        clen = take - cpos
                        if clen > CHUNK_C:
                            clen = CHUNK_C
                        tle.dsa.copy(
                            WorkGM + ((soff + cpos) * 2 + tl.arange(0, CP2)),
                            tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                            [clen * 2],
                        )
                        # copy writes the whole [CP2] block: when the block is
                        # full (clen==CHUNK_C) the block end == logical end,
                        # so copy is safe; a tail block would overrun and
                        # clobber the next segment, so take the masked-store
                        # path.
                        if clen == CHUNK_C:
                            tle.dsa.copy(
                                tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                                WorkGM
                                + (
                                    (dst_base + dst_seg_off + produced + cpos) * 2
                                    + tl.arange(0, CP2)
                                ),
                                [clen * 2],
                            )
                        else:
                            sv = tle.dsa.to_tensor(
                                tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1])
                            )
                            tl.store(
                                WorkGM
                                + (
                                    (dst_base + dst_seg_off + produced + cpos) * 2
                                    + tl.arange(0, CP2)
                                ),
                                sv,
                                mask=tl.arange(0, CP2) < clen * 2,
                            )
                        cpos += clen
                    produced += take
                else:
                    if l0 > 0:
                        tle.dsa.copy(
                            WorkGM + ((src_base + g0 + c0) * 2 + tl.arange(0, CP2)),
                            tle.dsa.subview(
                                in_ub, [0 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                            ),
                            [l0 * 2],
                        )
                    if l1 > 0:
                        tle.dsa.copy(
                            WorkGM + ((src_base + g1 + c1) * 2 + tl.arange(0, CP2)),
                            tle.dsa.subview(
                                in_ub, [1 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                            ),
                            [l1 * 2],
                        )
                    if l2 > 0:
                        tle.dsa.copy(
                            WorkGM + ((src_base + g2 + c2) * 2 + tl.arange(0, CP2)),
                            tle.dsa.subview(
                                in_ub, [2 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                            ),
                            [l2 * 2],
                        )
                    if l3 > 0:
                        tle.dsa.copy(
                            WorkGM + ((src_base + g3 + c3) * 2 + tl.arange(0, CP2)),
                            tle.dsa.subview(
                                in_ub, [3 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                            ),
                            [l3 * 2],
                        )

                    _, cons = tle.dsa.ascend.raw(
                        "merge_exhaust_sort4",
                        tle.dsa.to_tensor(in_ub),
                        aw,
                        0 * CHUNK_C,
                        1 * CHUNK_C,
                        2 * CHUNK_C,
                        3 * CHUNK_C,
                        l0,
                        l1,
                        l2,
                        l3,
                        out=[tle.dsa.to_tensor(out_ub), cons],
                    )

                    e0 = al.get_element(cons, (0,))
                    e1 = al.get_element(cons, (1,))
                    e2 = al.get_element(cons, (2,))
                    e3 = al.get_element(cons, (3,))
                    batch = e0 + e1 + e2 + e3

                    take = batch
                    if produced + take > grp_cap:
                        take = grp_cap - produced
                    ot = tle.dsa.to_tensor(tle.dsa.subview(out_ub, [0], [OP2], [1]))
                    tl.store(
                        WorkGM
                        + ((dst_base + dst_seg_off + produced) * 2 + tl.arange(0, OP2)),
                        ot,
                        mask=tl.arange(0, OP2) < take * 2,
                    )
                    c0 += e0
                    c1 += e1
                    c2 += e2
                    c3 += e3
                    produced += batch
                    if batch == 0:
                        produced = grp_cap

        cur_n = out_n
        cur_full = nxt_full
        cur_last = nxt_last
        if src_phase == 0:
            src_phase = 1
        else:
            src_phase = 0

    offs = tl.arange(0, UNPACK_CHUNK)
    if src_phase == 1:
        dval = tl.zeros([UNPACK_CHUNK], dtype=tl.float32)
        didx = tl.zeros([UNPACK_CHUNK], dtype=tl.int32)
        for c in tl.range(0, NUM_CHUNKS):
            cstart = c * UNPACK_CHUNK
            clen = K - cstart
            if clen > UNPACK_CHUNK:
                clen = UNPACK_CHUNK
            tle.dsa.copy(
                WorkGM + ((base_p1 + cstart) * 2 + tl.arange(0, UP2)),
                tle.dsa.subview(in_ub, [0], [UP2], [1]),
                [clen * 2],
            )
            dval, didx = tle.dsa.ascend.raw(
                "unpack_sort", tle.dsa.to_tensor(in_ub), clen, out=[dval, didx]
            )
            tl.store(Yv + row * K + cstart + offs, dval, mask=offs < clen)
            tl.store(Yi + row * K + cstart + offs, didx, mask=offs < clen)
    else:
        dval = tl.zeros([UNPACK_CHUNK], dtype=tl.float32)
        didx = tl.zeros([UNPACK_CHUNK], dtype=tl.int32)
        for c in tl.range(0, NUM_CHUNKS):
            cstart = c * UNPACK_CHUNK
            clen = K - cstart
            if clen > UNPACK_CHUNK:
                clen = UNPACK_CHUNK
            tle.dsa.copy(
                WorkGM + ((base_p0 + cstart) * 2 + tl.arange(0, UP2)),
                tle.dsa.subview(in_ub, [0], [UP2], [1]),
                [clen * 2],
            )
            dval, didx = tle.dsa.ascend.raw(
                "unpack_sort", tle.dsa.to_tensor(in_ub), clen, out=[dval, didx]
            )
            tl.store(Yv + row * K + cstart + offs, dval, mask=offs < clen)
            tl.store(Yi + row * K + cstart + offs, didx, mask=offs < clen)


@triton.jit
def smallk_stream_merge_unpack_kernel(
    WorkGM,
    Yv,
    Yi,
    ROW_WORDS: tl.constexpr,
    RUN_LEN: tl.constexpr,
    NUM_RUNS_C: tl.constexpr,
    K: tl.constexpr,
    STREAM_CAP: tl.constexpr,
    UNPACK_CAP: tl.constexpr,
):
    # Only dispatched with NUM_RUNS_C > 4 (== 4 goes to the merge4 kernel).
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    src_base = (row * 2 + 0) * region_props

    cons = tl.zeros([4], dtype=tl.int32)
    a_ub = tle.dsa.alloc(
        [STREAM_CAP], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    b_ub = tle.dsa.alloc(
        [STREAM_CAP], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    # Both sides of a copy must have equal static shapes: the per-way copy is
    # aligned to npo2(RUN_LEN*2), so arange/subview also use W2W; the extra
    # out-of-bounds data read past the end is ignored by the asm length.
    W2W: tl.constexpr = triton.next_power_of_2(RUN_LEN * 2)

    first_runs = NUM_RUNS_C
    if first_runs > 4:
        first_runs = 4

    tle.dsa.copy(
        WorkGM + (src_base * 2 + tl.arange(0, STREAM_CAP)),
        a_ub,
        [first_runs * RUN_LEN * 2],
    )

    phase = 0
    if first_runs > 1:
        l0 = RUN_LEN
        l1 = RUN_LEN if first_runs > 1 else 0
        l2 = RUN_LEN if first_runs > 2 else 0
        l3 = RUN_LEN if first_runs > 3 else 0
        tle.dsa.ascend.raw(
            "merge_exhaust_sort4",
            tle.dsa.to_tensor(a_ub),
            4,
            0,
            RUN_LEN,
            2 * RUN_LEN,
            3 * RUN_LEN,
            l0,
            l1,
            l2,
            l3,
            out=[tle.dsa.to_tensor(b_ub), cons],
        )
        phase = 1

    run = first_runs
    while run < NUM_RUNS_C:
        next_runs = NUM_RUNS_C - run
        if next_runs > 3:
            next_runs = 3

        if phase == 0:
            for j in tl.static_range(3):
                tle.dsa.copy(
                    WorkGM
                    + (((src_base + (run + j) * RUN_LEN) * 2) + tl.arange(0, W2W)),
                    tle.dsa.subview(a_ub, [(1 + j) * RUN_LEN * 2], [W2W], [1]),
                    [RUN_LEN * 2],
                )
            l0 = RUN_LEN
            l1 = RUN_LEN if next_runs > 0 else 0
            l2 = RUN_LEN if next_runs > 1 else 0
            l3 = RUN_LEN if next_runs > 2 else 0
            tle.dsa.ascend.raw(
                "merge_exhaust_sort4",
                tle.dsa.to_tensor(a_ub),
                4,
                0,
                RUN_LEN,
                2 * RUN_LEN,
                3 * RUN_LEN,
                l0,
                l1,
                l2,
                l3,
                out=[tle.dsa.to_tensor(b_ub), cons],
            )
            phase = 1
        else:
            for j in tl.static_range(3):
                tle.dsa.copy(
                    WorkGM
                    + (((src_base + (run + j) * RUN_LEN) * 2) + tl.arange(0, W2W)),
                    tle.dsa.subview(b_ub, [(1 + j) * RUN_LEN * 2], [W2W], [1]),
                    [RUN_LEN * 2],
                )
            l0 = RUN_LEN
            l1 = RUN_LEN if next_runs > 0 else 0
            l2 = RUN_LEN if next_runs > 1 else 0
            l3 = RUN_LEN if next_runs > 2 else 0
            tle.dsa.ascend.raw(
                "merge_exhaust_sort4",
                tle.dsa.to_tensor(b_ub),
                4,
                0,
                RUN_LEN,
                2 * RUN_LEN,
                3 * RUN_LEN,
                l0,
                l1,
                l2,
                l3,
                out=[tle.dsa.to_tensor(a_ub), cons],
            )
            phase = 0
        run += next_runs

    dval = tl.zeros([UNPACK_CAP], dtype=tl.float32)
    didx = tl.zeros([UNPACK_CAP], dtype=tl.int32)
    offs = tl.arange(0, UNPACK_CAP)
    if phase == 0:
        dval, didx = tle.dsa.ascend.raw(
            "unpack_sort", tle.dsa.to_tensor(a_ub), K, out=[dval, didx]
        )
    else:
        dval, didx = tle.dsa.ascend.raw(
            "unpack_sort", tle.dsa.to_tensor(b_ub), K, out=[dval, didx]
        )
    tl.store(Yv + row * K + offs, dval, mask=offs < K)
    tl.store(Yi + row * K + offs, didx, mask=offs < K)


@triton.jit
def smallk_stream_merge4_unpack_kernel(
    WorkGM,
    Yv,
    Yi,
    ROW_WORDS: tl.constexpr,
    RUN_LEN: tl.constexpr,
    NUM_RUNS_C: tl.constexpr,
    K: tl.constexpr,
    STREAM_CAP: tl.constexpr,
    UNPACK_CAP: tl.constexpr,
):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    src_base = (row * 2 + 0) * region_props

    cons = tl.zeros([4], dtype=tl.int32)
    in_ub = tle.dsa.alloc(
        [STREAM_CAP], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    out_ub = tle.dsa.alloc(
        [STREAM_CAP], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )

    tle.dsa.copy(
        WorkGM + (src_base * 2 + tl.arange(0, STREAM_CAP)),
        in_ub,
        [NUM_RUNS_C * RUN_LEN * 2],
    )

    l0 = RUN_LEN
    l1 = RUN_LEN if NUM_RUNS_C > 1 else 0
    l2 = RUN_LEN if NUM_RUNS_C > 2 else 0
    l3 = RUN_LEN if NUM_RUNS_C > 3 else 0
    tle.dsa.ascend.raw(
        "merge_exhaust_sort4",
        tle.dsa.to_tensor(in_ub),
        4,
        0,
        RUN_LEN,
        2 * RUN_LEN,
        3 * RUN_LEN,
        l0,
        l1,
        l2,
        l3,
        out=[tle.dsa.to_tensor(out_ub), cons],
    )

    dval = tl.zeros([UNPACK_CAP], dtype=tl.float32)
    didx = tl.zeros([UNPACK_CAP], dtype=tl.int32)
    offs = tl.arange(0, UNPACK_CAP)
    dval, didx = tle.dsa.ascend.raw(
        "unpack_sort", tle.dsa.to_tensor(out_ub), K, out=[dval, didx]
    )
    tl.store(Yv + row * K + offs, dval, mask=offs < K)
    tl.store(Yi + row * K + offs, didx, mask=offs < K)


@triton.jit
def smallk_core_local_stream_merge_kernel(
    SortGM,
    LocalGM,
    NUM_SEG_PER_ROW: tl.constexpr,
    SORT_ROW_WORDS: tl.constexpr,
    LOCAL_ROW_WORDS: tl.constexpr,
    RUN_LEN: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES_C: tl.constexpr,
    STREAM_CAP: tl.constexpr,
):
    row = tl.program_id(0)
    cid = tl.program_id(1)

    sort_region_props = SORT_ROW_WORDS // 2
    local_region_props = LOCAL_ROW_WORDS // 2
    sort_base = (row * 2 + 0) * sort_region_props
    local_base = (row * 2 + 0) * local_region_props + cid * K

    first_seg = (cid * NUM_SEG_PER_ROW) // NUM_CORES_C
    end_seg = ((cid + 1) * NUM_SEG_PER_ROW) // NUM_CORES_C
    local_n = end_seg - first_seg
    src_base = sort_base + first_seg * RUN_LEN

    cons = tl.zeros([4], dtype=tl.int32)
    a_ub = tle.dsa.alloc(
        [STREAM_CAP], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    b_ub = tle.dsa.alloc(
        [STREAM_CAP], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    # Both sides of a copy must have equal static shapes: the per-way copy is
    # aligned to npo2(RUN_LEN*2), so arange/subview also use W2W; the extra
    # out-of-bounds data read past the end is ignored by the asm length.
    W2W: tl.constexpr = triton.next_power_of_2(RUN_LEN * 2)
    KP2: tl.constexpr = triton.next_power_of_2(K * 2)

    first_runs = local_n
    if first_runs > 4:
        first_runs = 4

    tle.dsa.copy(
        SortGM + (src_base * 2 + tl.arange(0, STREAM_CAP)),
        a_ub,
        [first_runs * RUN_LEN * 2],
    )

    phase = 0
    if first_runs > 1:
        l0 = RUN_LEN
        l1 = RUN_LEN if first_runs > 1 else 0
        l2 = RUN_LEN if first_runs > 2 else 0
        l3 = RUN_LEN if first_runs > 3 else 0
        tle.dsa.ascend.raw(
            "merge_exhaust_sort4",
            tle.dsa.to_tensor(a_ub),
            4,
            0,
            RUN_LEN,
            2 * RUN_LEN,
            3 * RUN_LEN,
            l0,
            l1,
            l2,
            l3,
            out=[tle.dsa.to_tensor(b_ub), cons],
        )
        phase = 1

    run = first_runs
    while run < local_n:
        next_runs = local_n - run
        if next_runs > 3:
            next_runs = 3

        if phase == 0:
            for j in tl.static_range(3):
                tle.dsa.copy(
                    SortGM
                    + (((src_base + (run + j) * RUN_LEN) * 2) + tl.arange(0, W2W)),
                    tle.dsa.subview(a_ub, [(1 + j) * RUN_LEN * 2], [W2W], [1]),
                    [RUN_LEN * 2],
                )
            l0 = RUN_LEN
            l1 = RUN_LEN if next_runs > 0 else 0
            l2 = RUN_LEN if next_runs > 1 else 0
            l3 = RUN_LEN if next_runs > 2 else 0
            tle.dsa.ascend.raw(
                "merge_exhaust_sort4",
                tle.dsa.to_tensor(a_ub),
                4,
                0,
                RUN_LEN,
                2 * RUN_LEN,
                3 * RUN_LEN,
                l0,
                l1,
                l2,
                l3,
                out=[tle.dsa.to_tensor(b_ub), cons],
            )
            phase = 1
        else:
            for j in tl.static_range(3):
                tle.dsa.copy(
                    SortGM
                    + (((src_base + (run + j) * RUN_LEN) * 2) + tl.arange(0, W2W)),
                    tle.dsa.subview(b_ub, [(1 + j) * RUN_LEN * 2], [W2W], [1]),
                    [RUN_LEN * 2],
                )
            l0 = RUN_LEN
            l1 = RUN_LEN if next_runs > 0 else 0
            l2 = RUN_LEN if next_runs > 1 else 0
            l3 = RUN_LEN if next_runs > 2 else 0
            tle.dsa.ascend.raw(
                "merge_exhaust_sort4",
                tle.dsa.to_tensor(b_ub),
                4,
                0,
                RUN_LEN,
                2 * RUN_LEN,
                3 * RUN_LEN,
                l0,
                l1,
                l2,
                l3,
                out=[tle.dsa.to_tensor(a_ub), cons],
            )
            phase = 0

        run += next_runs

    # When 2K is exactly a power of two the block end == logical end and copy
    # stays in bounds; otherwise use a masked store.
    if K * 2 == KP2:
        if phase == 0:
            tle.dsa.copy(
                tle.dsa.subview(a_ub, [0], [KP2], [1]),
                LocalGM + (local_base * 2 + tl.arange(0, KP2)),
                [K * 2],
            )
        else:
            tle.dsa.copy(
                tle.dsa.subview(b_ub, [0], [KP2], [1]),
                LocalGM + (local_base * 2 + tl.arange(0, KP2)),
                [K * 2],
            )
    else:
        dst_offs = tl.arange(0, STREAM_CAP)
        if phase == 0:
            tl.store(
                LocalGM + (local_base * 2 + dst_offs),
                tle.dsa.to_tensor(a_ub),
                mask=dst_offs < K * 2,
            )
        else:
            tl.store(
                LocalGM + (local_base * 2 + dst_offs),
                tle.dsa.to_tensor(b_ub),
                mask=dst_offs < K * 2,
            )


@triton.jit
def static_merge4_unpack_kernel(
    WorkGM,
    Yv,
    Yi,
    ROW_WORDS: tl.constexpr,
    RUN_LEN: tl.constexpr,
    NUM_RUNS_C: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    src_base = (row * 2 + 0) * region_props

    CP2: tl.constexpr = triton.next_power_of_2(NUM_RUNS_C * RUN_LEN * 2)
    in_ub = tle.dsa.alloc([CP2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    out_ub = tle.dsa.alloc([CP2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(
        WorkGM + (src_base * 2 + tl.arange(0, CP2)), in_ub, [NUM_RUNS_C * RUN_LEN * 2]
    )

    cons = tl.zeros([4], dtype=tl.int32)
    l0 = RUN_LEN
    l1 = RUN_LEN if NUM_RUNS_C > 1 else 0
    l2 = RUN_LEN if NUM_RUNS_C > 2 else 0
    l3 = RUN_LEN if NUM_RUNS_C > 3 else 0
    tle.dsa.ascend.raw(
        "merge_exhaust_sort4",
        tle.dsa.to_tensor(in_ub),
        NUM_RUNS_C,
        0,
        RUN_LEN,
        2 * RUN_LEN,
        3 * RUN_LEN,
        l0,
        l1,
        l2,
        l3,
        out=[tle.dsa.to_tensor(out_ub), cons],
    )

    KP2: tl.constexpr = triton.next_power_of_2(K)
    dval = tl.zeros([KP2], dtype=tl.float32)
    didx = tl.zeros([KP2], dtype=tl.int32)
    offs = tl.arange(0, KP2)
    dval, didx = tle.dsa.ascend.raw(
        "unpack_sort", tle.dsa.to_tensor(out_ub), K, out=[dval, didx]
    )
    tl.store(Yv + row * K + offs, dval, mask=offs < K)
    tl.store(Yi + row * K + offs, didx, mask=offs < K)


# ============================================================================
# Final batch-local standalone implementation
# ============================================================================
# This section inlines the batch scheduling experiment from
# topk-step-two-batch-core-local-merge.py.


@triton.jit
def core_local_nocopy_merge_kernel(
    SortGM,
    TmpGM,
    FinalGM,
    NUM_SEG_PER_ROW: tl.constexpr,
    SORT_RUN_LEN: tl.constexpr,
    SORT_ROW_WORDS: tl.constexpr,
    TMP_ROW_WORDS: tl.constexpr,
    FINAL_ROW_WORDS: tl.constexpr,
    CORE_WORK_LEN: tl.constexpr,
    CORE_OUT_LEN: tl.constexpr,
    K: tl.constexpr,
    CHUNK_C: tl.constexpr,
    IN_CAP: tl.constexpr,
    OUT_CAP: tl.constexpr,
    MAX_STREAM_ITERS: tl.constexpr,
    MAX_WAYS_C: tl.constexpr,
    LOCAL_GROUPS_C: tl.constexpr,
    NUM_CORES_C: tl.constexpr,
    SKIP_FINAL_COPY: tl.constexpr,
):
    row = tl.program_id(0)
    cid = tl.program_id(1)

    sort_region_props = SORT_ROW_WORDS // 2
    tmp_region_props = TMP_ROW_WORDS // 2
    final_region_props = FINAL_ROW_WORDS // 2

    sort_base = (row * 2 + 0) * sort_region_props
    tmp_base_p0 = (row * 2 + 0) * tmp_region_props + cid * CORE_WORK_LEN
    tmp_base_p1 = (row * 2 + 1) * tmp_region_props + cid * CORE_WORK_LEN
    final_base_p0 = (row * 2 + 0) * final_region_props + cid * CORE_OUT_LEN

    first_seg = (cid * NUM_SEG_PER_ROW) // NUM_CORES_C
    end_seg = ((cid + 1) * NUM_SEG_PER_ROW) // NUM_CORES_C
    local_n = end_seg - first_seg

    cons = tl.zeros([4], dtype=tl.int32)
    CP2: tl.constexpr = triton.next_power_of_2(CHUNK_C * 2)
    OP2: tl.constexpr = triton.next_power_of_2(OUT_CAP)
    in_ub = tle.dsa.alloc([IN_CAP], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    out_ub = tle.dsa.alloc(
        [OUT_CAP + 8], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )

    # First merge round reads directly from SortGM and writes TmpGM phase 1.
    # This removes the original full SortGM->TmpGM pre-copy.
    cur_n = local_n
    cur_full = SORT_RUN_LEN
    cur_last = SORT_RUN_LEN
    src_phase = 1

    stride = SORT_RUN_LEN
    out_len = SORT_RUN_LEN * MAX_WAYS_C
    if out_len > K:
        out_len = K
    out_n = (cur_n + MAX_WAYS_C - 1) // MAX_WAYS_C
    last_ways = cur_n - MAX_WAYS_C * (out_n - 1)
    last_total = (last_ways - 1) * SORT_RUN_LEN + SORT_RUN_LEN
    nxt_full = out_len
    nxt_last = last_total
    if nxt_last > K:
        nxt_last = K

    for g in tl.range(0, LOCAL_GROUPS_C):
        if g < out_n:
            base_seg = g * MAX_WAYS_C
            ways = MAX_WAYS_C
            if base_seg + ways > cur_n:
                ways = cur_n - base_seg
            is_last_group = g == out_n - 1
            g0 = (base_seg + 0) * stride
            g1 = (base_seg + 1) * stride
            g2 = (base_seg + 2) * stride
            g3 = (base_seg + 3) * stride
            sl0 = SORT_RUN_LEN
            sl1 = SORT_RUN_LEN if ways > 1 else 0
            sl2 = SORT_RUN_LEN if ways > 2 else 0
            sl3 = SORT_RUN_LEN if ways > 3 else 0
            if is_last_group:
                if ways == 1:
                    sl0 = cur_last
                elif ways == 2:
                    sl1 = cur_last
                elif ways == 3:
                    sl2 = cur_last
                else:
                    sl3 = cur_last

            total = sl0 + sl1 + sl2 + sl3
            grp_cap = total
            if grp_cap > K:
                grp_cap = K
            dst_seg_off = g * out_len

            c0 = 0
            c1 = 0
            c2 = 0
            c3 = 0
            produced = 0
            it = 0
            while (produced < grp_cap) and (it < MAX_STREAM_ITERS):
                it += 1
                r0 = sl0 - c0
                r1 = sl1 - c1 if ways > 1 else 0
                r2 = sl2 - c2 if ways > 2 else 0
                r3 = sl3 - c3 if ways > 3 else 0
                l0 = r0 if r0 < CHUNK_C else CHUNK_C
                l1 = r1 if r1 < CHUNK_C else CHUNK_C
                l2 = r2 if r2 < CHUNK_C else CHUNK_C
                l3 = r3 if r3 < CHUNK_C else CHUNK_C
                rem_ways = 0
                if r0 > 0:
                    rem_ways += 1
                if r1 > 0:
                    rem_ways += 1
                if r2 > 0:
                    rem_ways += 1
                if r3 > 0:
                    rem_ways += 1
                aw = 0
                if l0 > 0:
                    aw += 1
                if l1 > 0:
                    aw += 1
                if l2 > 0:
                    aw += 1
                if l3 > 0:
                    aw += 1

                if rem_ways == 0:
                    produced = grp_cap
                elif rem_ways == 1:
                    src_seg_idx = first_seg + base_seg
                    sres = r0
                    csrc = c0
                    if r1 > 0:
                        src_seg_idx = first_seg + base_seg + 1
                        sres = r1
                        csrc = c1
                    if r2 > 0:
                        src_seg_idx = first_seg + base_seg + 2
                        sres = r2
                        csrc = c2
                    if r3 > 0:
                        src_seg_idx = first_seg + base_seg + 3
                        sres = r3
                        csrc = c3
                    take = sres
                    if produced + take > grp_cap:
                        take = grp_cap - produced
                    cpos = 0
                    while cpos < take:
                        clen = take - cpos
                        if clen > CHUNK_C:
                            clen = CHUNK_C
                        tle.dsa.copy(
                            SortGM
                            + (
                                (sort_base + src_seg_idx * SORT_RUN_LEN + csrc + cpos)
                                * 2
                                + tl.arange(0, CP2)
                            ),
                            tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                            [clen * 2],
                        )
                        if clen == CHUNK_C:
                            tle.dsa.copy(
                                tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                                TmpGM
                                + (
                                    (tmp_base_p1 + dst_seg_off + produced + cpos) * 2
                                    + tl.arange(0, CP2)
                                ),
                                [clen * 2],
                            )
                        else:
                            sv = tle.dsa.to_tensor(
                                tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1])
                            )
                            tl.store(
                                TmpGM
                                + (
                                    (tmp_base_p1 + dst_seg_off + produced + cpos) * 2
                                    + tl.arange(0, CP2)
                                ),
                                sv,
                                mask=tl.arange(0, CP2) < clen * 2,
                            )
                        cpos += clen
                    produced += take
                else:
                    if l0 > 0:
                        tle.dsa.copy(
                            SortGM
                            + (
                                (
                                    sort_base
                                    + (first_seg + base_seg + 0) * SORT_RUN_LEN
                                    + c0
                                )
                                * 2
                                + tl.arange(0, CP2)
                            ),
                            tle.dsa.subview(
                                in_ub, [0 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                            ),
                            [l0 * 2],
                        )
                    if l1 > 0:
                        tle.dsa.copy(
                            SortGM
                            + (
                                (
                                    sort_base
                                    + (first_seg + base_seg + 1) * SORT_RUN_LEN
                                    + c1
                                )
                                * 2
                                + tl.arange(0, CP2)
                            ),
                            tle.dsa.subview(
                                in_ub, [1 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                            ),
                            [l1 * 2],
                        )
                    if l2 > 0:
                        tle.dsa.copy(
                            SortGM
                            + (
                                (
                                    sort_base
                                    + (first_seg + base_seg + 2) * SORT_RUN_LEN
                                    + c2
                                )
                                * 2
                                + tl.arange(0, CP2)
                            ),
                            tle.dsa.subview(
                                in_ub, [2 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                            ),
                            [l2 * 2],
                        )
                    if l3 > 0:
                        tle.dsa.copy(
                            SortGM
                            + (
                                (
                                    sort_base
                                    + (first_seg + base_seg + 3) * SORT_RUN_LEN
                                    + c3
                                )
                                * 2
                                + tl.arange(0, CP2)
                            ),
                            tle.dsa.subview(
                                in_ub, [3 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                            ),
                            [l3 * 2],
                        )

                    _, cons = tle.dsa.ascend.raw(
                        "merge_exhaust_sort4",
                        tle.dsa.to_tensor(in_ub),
                        aw,
                        0 * CHUNK_C,
                        1 * CHUNK_C,
                        2 * CHUNK_C,
                        3 * CHUNK_C,
                        l0,
                        l1,
                        l2,
                        l3,
                        out=[tle.dsa.to_tensor(out_ub), cons],
                    )

                    e0 = al.get_element(cons, (0,))
                    e1 = al.get_element(cons, (1,))
                    e2 = al.get_element(cons, (2,))
                    e3 = al.get_element(cons, (3,))
                    batch = e0 + e1 + e2 + e3
                    take = batch
                    if produced + take > grp_cap:
                        take = grp_cap - produced
                    ot = tle.dsa.to_tensor(tle.dsa.subview(out_ub, [0], [OP2], [1]))
                    tl.store(
                        TmpGM
                        + (
                            (tmp_base_p1 + dst_seg_off + produced) * 2
                            + tl.arange(0, OP2)
                        ),
                        ot,
                        mask=tl.arange(0, OP2) < take * 2,
                    )
                    c0 += e0
                    c1 += e1
                    c2 += e2
                    c3 += e3
                    produced += batch
                    if batch == 0:
                        produced = grp_cap

    cur_n = out_n
    cur_full = nxt_full
    cur_last = nxt_last

    while cur_n > 1:
        stride = cur_full
        out_len = cur_full * MAX_WAYS_C
        if out_len > K:
            out_len = K
        out_n = (cur_n + MAX_WAYS_C - 1) // MAX_WAYS_C
        last_ways = cur_n - MAX_WAYS_C * (out_n - 1)
        last_total = (last_ways - 1) * cur_full + cur_last
        nxt_full = out_len
        nxt_last = last_total
        if nxt_last > K:
            nxt_last = K

        src_base = tmp_base_p0
        if src_phase == 1:
            src_base = tmp_base_p1
        dst_base = tmp_base_p1
        if src_phase == 1:
            dst_base = tmp_base_p0

        for g in tl.range(0, LOCAL_GROUPS_C):
            if g < out_n:
                base_seg = g * MAX_WAYS_C
                ways = MAX_WAYS_C
                if base_seg + ways > cur_n:
                    ways = cur_n - base_seg
                is_last_group = g == out_n - 1
                g0 = (base_seg + 0) * stride
                g1 = (base_seg + 1) * stride
                g2 = (base_seg + 2) * stride
                g3 = (base_seg + 3) * stride
                sl0 = cur_full
                sl1 = cur_full if ways > 1 else 0
                sl2 = cur_full if ways > 2 else 0
                sl3 = cur_full if ways > 3 else 0
                if is_last_group:
                    if ways == 1:
                        sl0 = cur_last
                    elif ways == 2:
                        sl1 = cur_last
                    elif ways == 3:
                        sl2 = cur_last
                    else:
                        sl3 = cur_last
                total = sl0 + sl1 + sl2 + sl3
                grp_cap = total
                if grp_cap > K:
                    grp_cap = K
                dst_seg_off = g * out_len

                c0 = 0
                c1 = 0
                c2 = 0
                c3 = 0
                produced = 0
                it = 0
                while (produced < grp_cap) and (it < MAX_STREAM_ITERS):
                    it += 1
                    r0 = sl0 - c0
                    r1 = sl1 - c1 if ways > 1 else 0
                    r2 = sl2 - c2 if ways > 2 else 0
                    r3 = sl3 - c3 if ways > 3 else 0
                    l0 = r0 if r0 < CHUNK_C else CHUNK_C
                    l1 = r1 if r1 < CHUNK_C else CHUNK_C
                    l2 = r2 if r2 < CHUNK_C else CHUNK_C
                    l3 = r3 if r3 < CHUNK_C else CHUNK_C
                    rem_ways = 0
                    if r0 > 0:
                        rem_ways += 1
                    if r1 > 0:
                        rem_ways += 1
                    if r2 > 0:
                        rem_ways += 1
                    if r3 > 0:
                        rem_ways += 1
                    aw = 0
                    if l0 > 0:
                        aw += 1
                    if l1 > 0:
                        aw += 1
                    if l2 > 0:
                        aw += 1
                    if l3 > 0:
                        aw += 1

                    if rem_ways == 0:
                        produced = grp_cap
                    elif rem_ways == 1:
                        soff = src_base + g0 + c0
                        sres = r0
                        if r1 > 0:
                            soff = src_base + g1 + c1
                            sres = r1
                        if r2 > 0:
                            soff = src_base + g2 + c2
                            sres = r2
                        if r3 > 0:
                            soff = src_base + g3 + c3
                            sres = r3
                        take = sres
                        if produced + take > grp_cap:
                            take = grp_cap - produced
                        cpos = 0
                        while cpos < take:
                            clen = take - cpos
                            if clen > CHUNK_C:
                                clen = CHUNK_C
                            tle.dsa.copy(
                                TmpGM + ((soff + cpos) * 2 + tl.arange(0, CP2)),
                                tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                                [clen * 2],
                            )
                            if clen == CHUNK_C:
                                tle.dsa.copy(
                                    tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                                    TmpGM
                                    + (
                                        (dst_base + dst_seg_off + produced + cpos) * 2
                                        + tl.arange(0, CP2)
                                    ),
                                    [clen * 2],
                                )
                            else:
                                sv = tle.dsa.to_tensor(
                                    tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1])
                                )
                                tl.store(
                                    TmpGM
                                    + (
                                        (dst_base + dst_seg_off + produced + cpos) * 2
                                        + tl.arange(0, CP2)
                                    ),
                                    sv,
                                    mask=tl.arange(0, CP2) < clen * 2,
                                )
                            cpos += clen
                        produced += take
                    else:
                        if l0 > 0:
                            tle.dsa.copy(
                                TmpGM + ((src_base + g0 + c0) * 2 + tl.arange(0, CP2)),
                                tle.dsa.subview(
                                    in_ub, [0 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                                ),
                                [l0 * 2],
                            )
                        if l1 > 0:
                            tle.dsa.copy(
                                TmpGM + ((src_base + g1 + c1) * 2 + tl.arange(0, CP2)),
                                tle.dsa.subview(
                                    in_ub, [1 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                                ),
                                [l1 * 2],
                            )
                        if l2 > 0:
                            tle.dsa.copy(
                                TmpGM + ((src_base + g2 + c2) * 2 + tl.arange(0, CP2)),
                                tle.dsa.subview(
                                    in_ub, [2 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                                ),
                                [l2 * 2],
                            )
                        if l3 > 0:
                            tle.dsa.copy(
                                TmpGM + ((src_base + g3 + c3) * 2 + tl.arange(0, CP2)),
                                tle.dsa.subview(
                                    in_ub, [3 * CHUNK_C * 2], [CHUNK_C * 2], [1]
                                ),
                                [l3 * 2],
                            )

                        _, cons = tle.dsa.ascend.raw(
                            "merge_exhaust_sort4",
                            tle.dsa.to_tensor(in_ub),
                            aw,
                            0 * CHUNK_C,
                            1 * CHUNK_C,
                            2 * CHUNK_C,
                            3 * CHUNK_C,
                            l0,
                            l1,
                            l2,
                            l3,
                            out=[tle.dsa.to_tensor(out_ub), cons],
                        )

                        e0 = al.get_element(cons, (0,))
                        e1 = al.get_element(cons, (1,))
                        e2 = al.get_element(cons, (2,))
                        e3 = al.get_element(cons, (3,))
                        batch = e0 + e1 + e2 + e3
                        take = batch
                        if produced + take > grp_cap:
                            take = grp_cap - produced
                        ot = tle.dsa.to_tensor(tle.dsa.subview(out_ub, [0], [OP2], [1]))
                        tl.store(
                            TmpGM
                            + (
                                (dst_base + dst_seg_off + produced) * 2
                                + tl.arange(0, OP2)
                            ),
                            ot,
                            mask=tl.arange(0, OP2) < take * 2,
                        )
                        c0 += e0
                        c1 += e1
                        c2 += e2
                        c3 += e3
                        produced += batch
                        if batch == 0:
                            produced = grp_cap

        cur_n = out_n
        cur_full = nxt_full
        cur_last = nxt_last
        if src_phase == 0:
            src_phase = 1
        else:
            src_phase = 0

    if not SKIP_FINAL_COPY:
        valid_len = local_n * SORT_RUN_LEN
        if valid_len > K:
            valid_len = K
        if valid_len > CORE_OUT_LEN:
            valid_len = CORE_OUT_LEN

        if src_phase == 1:
            copy_pos = 0
            while copy_pos < valid_len:
                clen = valid_len - copy_pos
                if clen > CHUNK_C:
                    clen = CHUNK_C
                tle.dsa.copy(
                    TmpGM + ((tmp_base_p1 + copy_pos) * 2 + tl.arange(0, CP2)),
                    tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                    [clen * 2],
                )
                if clen == CHUNK_C:
                    tle.dsa.copy(
                        tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                        FinalGM + ((final_base_p0 + copy_pos) * 2 + tl.arange(0, CP2)),
                        [clen * 2],
                    )
                else:
                    sv = tle.dsa.to_tensor(
                        tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1])
                    )
                    tl.store(
                        FinalGM + ((final_base_p0 + copy_pos) * 2 + tl.arange(0, CP2)),
                        sv,
                        mask=tl.arange(0, CP2) < clen * 2,
                    )
                copy_pos += clen
        else:
            copy_pos = 0
            while copy_pos < valid_len:
                clen = valid_len - copy_pos
                if clen > CHUNK_C:
                    clen = CHUNK_C
                tle.dsa.copy(
                    TmpGM + ((tmp_base_p0 + copy_pos) * 2 + tl.arange(0, CP2)),
                    tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                    [clen * 2],
                )
                if clen == CHUNK_C:
                    tle.dsa.copy(
                        tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1]),
                        FinalGM + ((final_base_p0 + copy_pos) * 2 + tl.arange(0, CP2)),
                        [clen * 2],
                    )
                else:
                    sv = tle.dsa.to_tensor(
                        tle.dsa.subview(in_ub, [0], [CHUNK_C * 2], [1])
                    )
                    tl.store(
                        FinalGM + ((final_base_p0 + copy_pos) * 2 + tl.arange(0, CP2)),
                        sv,
                        mask=tl.arange(0, CP2) < clen * 2,
                    )
                copy_pos += clen

        offs = tl.arange(0, CHUNK_C)
        pad_pos = valid_len
        while pad_pos < CORE_OUT_LEN:
            plen = CORE_OUT_LEN - pad_pos
            if plen > CHUNK_C:
                plen = CHUNK_C
            tl.store(
                FinalGM + (final_base_p0 + pad_pos + offs) * 2,
                float("-inf"),
                mask=offs < plen,
            )
            tl.store(
                FinalGM + (final_base_p0 + pad_pos + offs) * 2 + 1,
                0.0,
                mask=offs < plen,
            )
            pad_pos += plen


SMALLK_SORT_CHUNK = 1024
SMALLK_SORT_WAYS = 4


@triton.jit
def sort_kernel(
    X,
    WorkGM,
    N_COLS: tl.constexpr,
    SEG_LEN: tl.constexpr,
    NUM_SEG: tl.constexpr,
    NUM_SEG_PER_ROW: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    SEGS_PER_CORE: tl.constexpr,
    NUM_CORES_C: tl.constexpr,
    TMP_SIZE: tl.constexpr,
    SORT_TOPK: tl.constexpr,
    PROP_SEG: tl.constexpr,
    SORT_IMPL: tl.constexpr,
):
    cid = tl.program_id(0)
    tmp = tl.zeros([TMP_SIZE], dtype=tl.float32)
    neg_inf = float("-inf")
    SP2: tl.constexpr = triton.next_power_of_2(SEG_LEN)
    PP2: tl.constexpr = triton.next_power_of_2(PROP_SEG)
    buf_ub = tle.dsa.alloc(
        [SEG_LEN], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    # Both sides of a copy must have equal static shapes: seg_ub is allocated
    # with npo2(PROP_SEG); the extra tail is never written out (masked store
    # below) and sort_1d_pack only writes the first PROP_SEG elements.
    seg_ub = tle.dsa.alloc([PP2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    for j in tl.range(0, SEGS_PER_CORE):
        seg = cid + j * NUM_CORES_C
        if seg < NUM_SEG:
            row = seg // NUM_SEG_PER_ROW
            seg_in_row = seg - row * NUM_SEG_PER_ROW
            col_off = seg_in_row * SEG_LEN
            in_off = row * N_COLS + col_off
            seg_real = N_COLS - col_off
            if seg_real > SEG_LEN:
                seg_real = SEG_LEN

            if seg_real < SEG_LEN:
                soffs = tl.arange(0, SP2)
                buf = tl.load(X + in_off + soffs, mask=soffs < seg_real, other=neg_inf)
                tle.dsa.ascend.raw(
                    "sort_1d_pack",
                    buf,
                    tmp,
                    True,
                    SORT_TOPK,
                    col_off,
                    SORT_IMPL,
                    out=tle.dsa.to_tensor(seg_ub),
                )
            else:
                tle.dsa.copy(X + (in_off + tl.arange(0, SP2)), buf_ub, [SEG_LEN])
                buf = tle.dsa.to_tensor(buf_ub)
                tle.dsa.ascend.raw(
                    "sort_1d_pack",
                    buf,
                    tmp,
                    True,
                    SORT_TOPK,
                    col_off,
                    SORT_IMPL,
                    out=tle.dsa.to_tensor(seg_ub),
                )
            dst_off = (row * 2 + 0) * ROW_WORDS + seg_in_row * PROP_SEG
            # tle.dsa.copy requires equal static shapes on both sides (and
            # writes whole blocks). When PROP_SEG=2k is exactly a power of two
            # use copy directly; otherwise tl.arange can only span PP2, whose
            # tail holds uninitialized data, so fall back to a masked store
            # that writes only the valid part.
            if PROP_SEG == PP2:
                tle.dsa.copy(seg_ub, WorkGM + (dst_off + tl.arange(0, PP2)), [PROP_SEG])
            else:
                offs = tl.arange(0, PP2)
                seg_t = tle.dsa.to_tensor(seg_ub)
                tl.store(WorkGM + dst_off + offs, seg_t, mask=offs < PROP_SEG)


@triton.jit
def final_merge_unpack_kernel(
    WorkGM,
    Yv,
    Yi,
    ROW_WORDS: tl.constexpr,
    RUN_LEN: tl.constexpr,
    RUN_STRIDE: tl.constexpr,
    NUM_RUNS_C: tl.constexpr,
    SRC_PHASE: tl.constexpr,
    K: tl.constexpr,
    CHUNK_C: tl.constexpr,
    IN_CAP: tl.constexpr,
    OUT_CAP: tl.constexpr,
    MAX_STREAM_ITERS: tl.constexpr,
    UNPACK_CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    src_base = (row * 2 + SRC_PHASE) * region_props
    dst_phase = 1 - SRC_PHASE
    dst_base = (row * 2 + dst_phase) * region_props

    cons = tl.zeros([4], dtype=tl.int32)
    CP2: tl.constexpr = triton.next_power_of_2(CHUNK_C * 2)
    OP2: tl.constexpr = triton.next_power_of_2(OUT_CAP)
    UP2: tl.constexpr = triton.next_power_of_2(UNPACK_CHUNK * 2)
    in_ub = tle.dsa.alloc([IN_CAP], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    out_ub = tle.dsa.alloc(
        [OUT_CAP + 8], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    cp_ub = tle.dsa.alloc(
        [CHUNK_C * 2 + 16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )

    sl0 = RUN_LEN if NUM_RUNS_C > 0 else 0
    sl1 = RUN_LEN if NUM_RUNS_C > 1 else 0
    sl2 = RUN_LEN if NUM_RUNS_C > 2 else 0
    sl3 = RUN_LEN if NUM_RUNS_C > 3 else 0
    g0 = 0
    g1 = RUN_STRIDE
    g2 = RUN_STRIDE * 2
    g3 = RUN_STRIDE * 3

    grp_cap = RUN_LEN * NUM_RUNS_C
    if grp_cap > K:
        grp_cap = K

    c0 = 0
    c1 = 0
    c2 = 0
    c3 = 0
    produced = 0
    it = 0
    while (produced < grp_cap) and (it < MAX_STREAM_ITERS):
        it += 1
        r0 = sl0 - c0
        r1 = sl1 - c1 if NUM_RUNS_C > 1 else 0
        r2 = sl2 - c2 if NUM_RUNS_C > 2 else 0
        r3 = sl3 - c3 if NUM_RUNS_C > 3 else 0
        l0 = r0 if r0 < CHUNK_C else CHUNK_C
        l1 = r1 if r1 < CHUNK_C else CHUNK_C
        l2 = r2 if r2 < CHUNK_C else CHUNK_C
        l3 = r3 if r3 < CHUNK_C else CHUNK_C

        rem_ways = 0
        if r0 > 0:
            rem_ways += 1
        if r1 > 0:
            rem_ways += 1
        if r2 > 0:
            rem_ways += 1
        if r3 > 0:
            rem_ways += 1
        aw = 0
        if l0 > 0:
            aw += 1
        if l1 > 0:
            aw += 1
        if l2 > 0:
            aw += 1
        if l3 > 0:
            aw += 1

        if rem_ways == 0:
            produced = grp_cap
        elif rem_ways == 1:
            soff = src_base + g0 + c0
            sres = r0
            if r1 > 0:
                soff = src_base + g1 + c1
                sres = r1
            if r2 > 0:
                soff = src_base + g2 + c2
                sres = r2
            if r3 > 0:
                soff = src_base + g3 + c3
                sres = r3
            take = sres
            if produced + take > grp_cap:
                take = grp_cap - produced
            cpos = 0
            while cpos < take:
                clen = take - cpos
                if clen > CHUNK_C:
                    clen = CHUNK_C
                tle.dsa.copy(
                    WorkGM + ((soff + cpos) * 2 + tl.arange(0, CP2)),
                    tle.dsa.subview(cp_ub, [0], [CP2], [1]),
                    [clen * 2],
                )
                if clen == CHUNK_C:
                    tle.dsa.copy(
                        tle.dsa.subview(cp_ub, [0], [CP2], [1]),
                        WorkGM + ((dst_base + produced + cpos) * 2 + tl.arange(0, CP2)),
                        [clen * 2],
                    )
                else:
                    sv = tle.dsa.to_tensor(tle.dsa.subview(cp_ub, [0], [CP2], [1]))
                    tl.store(
                        WorkGM + ((dst_base + produced + cpos) * 2 + tl.arange(0, CP2)),
                        sv,
                        mask=tl.arange(0, CP2) < clen * 2,
                    )
                cpos += clen
            produced += take
        else:
            if l0 > 0:
                tle.dsa.copy(
                    WorkGM + ((src_base + g0 + c0) * 2 + tl.arange(0, CP2)),
                    tle.dsa.subview(in_ub, [0 * CHUNK_C * 2], [CHUNK_C * 2], [1]),
                    [l0 * 2],
                )
            if l1 > 0:
                tle.dsa.copy(
                    WorkGM + ((src_base + g1 + c1) * 2 + tl.arange(0, CP2)),
                    tle.dsa.subview(in_ub, [1 * CHUNK_C * 2], [CHUNK_C * 2], [1]),
                    [l1 * 2],
                )
            if l2 > 0:
                tle.dsa.copy(
                    WorkGM + ((src_base + g2 + c2) * 2 + tl.arange(0, CP2)),
                    tle.dsa.subview(in_ub, [2 * CHUNK_C * 2], [CHUNK_C * 2], [1]),
                    [l2 * 2],
                )
            if l3 > 0:
                tle.dsa.copy(
                    WorkGM + ((src_base + g3 + c3) * 2 + tl.arange(0, CP2)),
                    tle.dsa.subview(in_ub, [3 * CHUNK_C * 2], [CHUNK_C * 2], [1]),
                    [l3 * 2],
                )

            _, cons = tle.dsa.ascend.raw(
                "merge_exhaust_sort4",
                tle.dsa.to_tensor(in_ub),
                aw,
                0 * CHUNK_C,
                1 * CHUNK_C,
                2 * CHUNK_C,
                3 * CHUNK_C,
                l0,
                l1,
                l2,
                l3,
                out=[tle.dsa.to_tensor(out_ub), cons],
            )

            e0 = al.get_element(cons, (0,))
            e1 = al.get_element(cons, (1,))
            e2 = al.get_element(cons, (2,))
            e3 = al.get_element(cons, (3,))
            batch = e0 + e1 + e2 + e3
            take = batch
            if produced + take > grp_cap:
                take = grp_cap - produced
            ot = tle.dsa.to_tensor(tle.dsa.subview(out_ub, [0], [OP2], [1]))
            tl.store(
                WorkGM + ((dst_base + produced) * 2 + tl.arange(0, OP2)),
                ot,
                mask=tl.arange(0, OP2) < take * 2,
            )
            c0 += e0
            c1 += e1
            c2 += e2
            c3 += e3
            produced += batch
            if batch == 0:
                produced = grp_cap

    dval = tl.zeros([UNPACK_CHUNK], dtype=tl.float32)
    didx = tl.zeros([UNPACK_CHUNK], dtype=tl.int32)
    offs = tl.arange(0, UNPACK_CHUNK)
    un_ub = tle.dsa.alloc([UP2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    for c in tl.range(0, NUM_CHUNKS):
        cstart = c * UNPACK_CHUNK
        clen = K - cstart
        if clen > UNPACK_CHUNK:
            clen = UNPACK_CHUNK
        tle.dsa.copy(
            WorkGM + ((dst_base + cstart) * 2 + tl.arange(0, UP2)), un_ub, [clen * 2]
        )
        dval, didx = tle.dsa.ascend.raw(
            "unpack_sort", tle.dsa.to_tensor(un_ub), clen, out=[dval, didx]
        )
        tl.store(Yv + row * K + cstart + offs, dval, mask=offs < clen)
        tl.store(Yi + row * K + cstart + offs, didx, mask=offs < clen)


@triton.jit
def unpack_kernel(
    WorkGM,
    Yv,
    Yi,
    ROW_WORDS: tl.constexpr,
    SRC_PHASE: tl.constexpr,
    K: tl.constexpr,
    UNPACK_CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    res_base = (row * 2 + SRC_PHASE) * region_props

    UP2: tl.constexpr = triton.next_power_of_2(UNPACK_CHUNK * 2)
    s_ub = tle.dsa.alloc([UP2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    dval = tl.zeros([UNPACK_CHUNK], dtype=tl.float32)
    didx = tl.zeros([UNPACK_CHUNK], dtype=tl.int32)
    offs = tl.arange(0, UNPACK_CHUNK)

    for c in tl.range(0, NUM_CHUNKS):
        cstart = c * UNPACK_CHUNK
        clen = K - cstart
        if clen > UNPACK_CHUNK:
            clen = UNPACK_CHUNK
        tle.dsa.copy(
            WorkGM + ((res_base + cstart) * 2 + tl.arange(0, UP2)), s_ub, [clen * 2]
        )
        dval, didx = tle.dsa.ascend.raw(
            "unpack_sort", tle.dsa.to_tensor(s_ub), clen, out=[dval, didx]
        )
        tl.store(Yv + row * K + cstart + offs, dval, mask=offs < clen)
        tl.store(Yi + row * K + cstart + offs, didx, mask=offs < clen)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _final_phase(num_runs: int) -> int:
    phase = 0
    cur = num_runs
    while cur > 1:
        cur = _ceil_div(cur, MAX_WAYS)
        phase ^= 1
    return phase


def _prefer_base_medium_path(
    batch: int, num_seg_per_row: int, k: int, seg_len: int
) -> bool:
    if num_seg_per_row < DIRECT_GROUP_MIN_SEGS:
        return (
            _sort_impl(min(k, seg_len), seg_len)
            == tle.dsa.ascend.custom_ops.SORT_IMPL_BASE
        )
    if num_seg_per_row > DIRECT_GROUP_MAX_SEGS:
        return False
    if k <= seg_len:
        return False

    total_cols = num_seg_per_row * seg_len
    k_ratio = k / total_cols if total_cols > 0 else 1.0
    group_runs = _ceil_div(num_seg_per_row, MAX_WAYS)
    final_runs = _ceil_div(group_runs, MAX_WAYS) if group_runs > 8 else group_runs

    # For large-k medium-S cases, the verified base path is better when its
    # group tree naturally ends at a 4-way final merge.  The batch-local path
    # would otherwise spend most time re-merging long runs in core-local.
    return k_ratio >= 0.45 and batch >= NUM_CORES // 2 and final_runs == MAX_WAYS


def _out_segs(batch: int, num_seg_per_row: int, k: int, seg_len: int) -> int:
    max_runs = min(NUM_CORES, num_seg_per_row)
    if max_runs <= 1:
        return max_runs
    if batch == 2:
        return min(16, max_runs)
    if batch <= 8 and num_seg_per_row <= DIRECT_GROUP_MAX_SEGS and k <= seg_len:
        return min(MAX_WAYS, max_runs)
    if batch == 16 and num_seg_per_row == 16 and k == seg_len:
        return min(MAX_WAYS, max_runs)

    def score(runs: int):
        programs = batch * runs
        waves = _ceil_div(programs, NUM_CORES)
        idle_tail = (-programs) % NUM_CORES
        return (waves, idle_tail, runs)

    return min(range(1, max_runs + 1), key=score)


def _use_smallk_sort(k: int, seg_len: int) -> bool:
    return (
        seg_len == SMALLK_SORT_CHUNK * SMALLK_SORT_WAYS and 0 < k <= SMALLK_SORT_MAX_K
    )


def _use_layered_sort(k: int, seg_len: int) -> bool:
    if seg_len != 4096 or k <= 0 or k > LAYERED_SORT_MAX_K:
        return False
    # The 4x1024 small-k path is still better around K=512, and the original
    # base path remains better around K=1024 in the measured large-S cases.
    # Keep layered for early-stop tiny K and the 2048 fixed-tree path.
    return k <= 128 or k >= 2048


def _sort_impl(k: int, seg_len: int) -> int:
    if _use_layered_sort(k, seg_len):
        return tle.dsa.ascend.custom_ops.SORT_IMPL_S4096_K1_128_K2048
    if _use_smallk_sort(k, seg_len):
        return tle.dsa.ascend.custom_ops.SORT_IMPL_S4096_K129_512
    return tle.dsa.ascend.custom_ops.SORT_IMPL_BASE


def _sort_impl_for_round_robin(k: int, seg_len: int, segs_per_core: int) -> int:
    sort_impl = _sort_impl(k, seg_len)
    allow_layered_multi = k <= 128
    if (
        sort_impl == tle.dsa.ascend.custom_ops.SORT_IMPL_S4096_K1_128_K2048
        and segs_per_core > 2
        and not allow_layered_multi
    ):
        # The layered 4096 sort is right on the UB boundary.  When the
        # round-robin kernel unrolls three or more segment iterations per
        # program, BiSheng allocates enough extra local buffer to overflow UB.
        return tle.dsa.ascend.custom_ops.SORT_IMPL_BASE
    return sort_impl


def _sort_tmp_size(seg_len: int, sort_run_len: int, sort_impl: int) -> int:
    if sort_impl == tle.dsa.ascend.custom_ops.SORT_IMPL_BASE:
        return seg_len * 4
    if sort_impl == tle.dsa.ascend.custom_ops.SORT_IMPL_S4096_K1_128_K2048:
        props_a_f32 = seg_len * 2
        props_b_f32 = seg_len * 2
        group_buf_f32 = 4 * 512 * 2
        final_out_f32 = 2 * group_buf_f32 + 8
        return props_a_f32 + props_b_f32 + 2 * group_buf_f32 + final_out_f32
    candidates_f32 = SMALLK_SORT_WAYS * sort_run_len * 2
    chunk_tmp_f32 = SMALLK_SORT_CHUNK * 4
    merge_out_f32 = SMALLK_SORT_WAYS * sort_run_len * 2
    return candidates_f32 + chunk_tmp_f32 + merge_out_f32


def _use_stream16_smallk_final(k: int, run_len: int, num_runs: int) -> bool:
    return (
        0 < k <= STREAM16_SMALLK_MERGE_MAX_K
        and run_len == k
        and num_runs > 1
        and (num_runs > STREAM16_SMALLK_GROUP_RUNS or k <= TINY_S_SMALL_BATCH_MAX_K)
    )


def _use_strided_local_final_merge(
    out_segs: int, num_seg_per_row: int, k: int, core_out_len: int, sort_run_len: int
) -> bool:
    # Low-risk subset: every local run yields the same logical output length.
    # Exact segment divisibility is sufficient.  For ragged splits, it is also
    # safe when the shortest local run still has >=K candidates, because every
    # local result is truncated to K before the final merge.
    min_input_segs_per_out_seg = num_seg_per_row // out_segs if out_segs > 0 else 0
    full_length_ragged = min_input_segs_per_out_seg * sort_run_len >= k
    phases = {
        _final_phase(
            max(
                2,
                ((cid + 1) * num_seg_per_row) // out_segs
                - (cid * num_seg_per_row) // out_segs,
            )
        )
        for cid in range(out_segs)
    }
    return (
        1 < out_segs <= MAX_WAYS
        and core_out_len == k
        and (num_seg_per_row % out_segs == 0 or full_length_ragged)
        and len(phases) == 1
    )


def _use_single_run_tmp_unpack(out_segs: int, k: int, core_out_len: int) -> bool:
    return out_segs == 1 and core_out_len == k


def _use_smalln_2048_path(batch: int, n: int, seg_len: int) -> bool:
    new_segments = _ceil_div(n, 2048)
    return (
        seg_len == 4096
        and SMALLN_2048_MIN_N <= n <= SMALLN_2048_MAX_N
        and new_segments <= MAX_WAYS
        and batch * new_segments <= NUM_CORES
    )


def _use_tiny_s_small_batch_path(batch: int, n: int, k: int, seg_len: int) -> bool:
    num_seg_per_row = _ceil_div(n, seg_len)
    small_segments = _ceil_div(n, 2048)
    return (
        seg_len == 4096
        and 0 < batch <= 4
        and 0 < k <= TINY_S_SMALL_BATCH_MAX_K
        and num_seg_per_row <= 2
        and small_segments <= MAX_WAYS
        and batch * small_segments <= NUM_CORES
    )


def _new_topk_outputs(x: torch.Tensor, batch: int, k: int):
    y_vals = torch.empty((batch, k), device=x.device, dtype=torch.float32)
    y_idx = torch.empty((batch, k), device=x.device, dtype=torch.int32)
    return y_vals, y_idx


def _build_sorted_runs_round_robin(x: torch.Tensor, k: int, *, seg_len: int):
    m, n = x.shape
    num_seg_per_row = _ceil_div(n, seg_len)
    num_seg = m * num_seg_per_row
    sort_run_len = k if k < seg_len else seg_len
    prop_seg = sort_run_len * 2
    sort_row_words = num_seg_per_row * prop_seg
    sort_words = m * 2 * sort_row_words
    sort_gm = torch.empty((sort_words,), device=x.device, dtype=torch.float32)
    x_flat = x.contiguous().view(-1)

    sort_grid = min(num_seg, NUM_CORES)
    segs_per_core = _ceil_div(num_seg, sort_grid)
    sort_impl = _sort_impl_for_round_robin(sort_run_len, seg_len, segs_per_core)
    tmp_size = _sort_tmp_size(seg_len, sort_run_len, sort_impl)
    sort_kernel[(sort_grid,)](
        x_flat,
        sort_gm,
        N_COLS=n,
        SEG_LEN=seg_len,
        NUM_SEG=num_seg,
        NUM_SEG_PER_ROW=num_seg_per_row,
        ROW_WORDS=sort_row_words,
        SEGS_PER_CORE=segs_per_core,
        NUM_CORES_C=sort_grid,
        TMP_SIZE=tmp_size,
        SORT_TOPK=sort_run_len,
        PROP_SEG=prop_seg,
        SORT_IMPL=sort_impl,
        multibuffer=False,
    )

    return sort_gm, sort_row_words, sort_run_len, num_seg_per_row


def _launch_merge_unpack(
    src_gm,
    y_vals,
    y_idx,
    *,
    batch: int,
    num_runs: int,
    run_len: int,
    row_words: int,
    k: int,
    chunk: int,
    unpack_chunk: int,
):
    num_chunks = _ceil_div(k, unpack_chunk)
    in_cap = MAX_WAYS * chunk * 2
    out_cap = MAX_WAYS * chunk * 2
    max_stream = (num_runs * run_len) // chunk + 8
    generic_merge_unpack_kernel[(batch,)](
        src_gm,
        y_vals,
        y_idx,
        NUM_SEG=num_runs,
        SEG_LEN=run_len,
        ROW_WORDS=row_words,
        K=k,
        CHUNK_C=chunk,
        IN_CAP=in_cap,
        OUT_CAP=out_cap,
        MAX_STREAM_ITERS=max_stream,
        MAX_WAYS_C=MAX_WAYS,
        UNPACK_CHUNK=unpack_chunk,
        NUM_CHUNKS=num_chunks,
        multibuffer=False,
        enable_select_analysis=False,
    )


def _launch_final_merge_unpack(
    src_gm,
    y_vals,
    y_idx,
    *,
    batch: int,
    num_runs: int,
    run_len: int,
    row_words: int,
    k: int,
    chunk: int,
    unpack_chunk: int,
    run_stride: int | None = None,
    src_phase: int = 0,
):
    if num_runs == 1:
        # A single run is already sorted: unpack directly. Also works around
        # an upstream MLIR compilation failure of final_merge_unpack_kernel
        # when NUM_RUNS_C == 1 (present in the original FlagTree source too).
        num_chunks = _ceil_div(k, unpack_chunk)
        unpack_kernel[(batch,)](
            src_gm,
            y_vals,
            y_idx,
            ROW_WORDS=row_words,
            SRC_PHASE=src_phase,
            K=k,
            UNPACK_CHUNK=unpack_chunk,
            NUM_CHUNKS=num_chunks,
            multibuffer=False,
            enable_select_analysis=False,
        )
        return

    if num_runs <= MAX_WAYS:
        if run_stride is None:
            run_stride = run_len
        num_chunks = _ceil_div(k, unpack_chunk)
        in_cap = MAX_WAYS * chunk * 2
        out_cap = MAX_WAYS * chunk * 2
        max_stream = (num_runs * run_len) // chunk + 8
        final_merge_unpack_kernel[(batch,)](
            src_gm,
            y_vals,
            y_idx,
            ROW_WORDS=row_words,
            RUN_LEN=run_len,
            RUN_STRIDE=run_stride,
            NUM_RUNS_C=num_runs,
            SRC_PHASE=src_phase,
            K=k,
            CHUNK_C=chunk,
            IN_CAP=in_cap,
            OUT_CAP=out_cap,
            MAX_STREAM_ITERS=max_stream,
            UNPACK_CHUNK=unpack_chunk,
            NUM_CHUNKS=num_chunks,
            multibuffer=False,
            enable_select_analysis=False,
        )
        return

    _launch_merge_unpack(
        src_gm,
        y_vals,
        y_idx,
        batch=batch,
        num_runs=num_runs,
        run_len=run_len,
        row_words=row_words,
        k=k,
        chunk=chunk,
        unpack_chunk=unpack_chunk,
    )


def _finish_with_final_merge_unpack(
    x: torch.Tensor,
    src_gm,
    *,
    k: int,
    num_runs: int,
    run_len: int,
    row_words: int,
    run_stride: int | None = None,
    src_phase: int = 0,
    y_vals=None,
    y_idx=None,
):
    m = x.shape[0]
    if y_vals is None or y_idx is None:
        y_vals, y_idx = _new_topk_outputs(x, m, k)
    _launch_final_merge_unpack(
        src_gm,
        y_vals,
        y_idx,
        batch=m,
        num_runs=num_runs,
        run_len=run_len,
        row_words=row_words,
        k=k,
        chunk=CHUNK,
        unpack_chunk=triton.next_power_of_2(min(k, CHUNK)),
        run_stride=run_stride,
        src_phase=src_phase,
    )
    return y_vals, y_idx


def _topk_sort_then_final_merge(x: torch.Tensor, k: int, *, seg_len: int):
    sort_gm, sort_row_words, sort_run_len, num_seg_per_row = (
        _build_sorted_runs_round_robin(x, k, seg_len=seg_len)
    )
    return _finish_with_final_merge_unpack(
        x,
        sort_gm,
        k=k,
        num_runs=num_seg_per_row,
        run_len=sort_run_len,
        row_words=sort_row_words,
    )


def _topk_tiny_s_small_batch_path(x: torch.Tensor, k: int):
    m, n = x.shape
    small_seg_len = 2048
    num_seg_per_row = _ceil_div(n, small_seg_len)
    num_seg = m * num_seg_per_row
    sort_run_len = k if k < small_seg_len else small_seg_len
    prop_seg = sort_run_len * 2
    x_flat = x.contiguous().view(-1)

    sort_row_words = num_seg_per_row * prop_seg
    sort_words = m * 2 * sort_row_words
    sort_gm = torch.empty((sort_words,), device=x.device, dtype=torch.float32)

    sort_impl = tle.dsa.ascend.custom_ops.SORT_IMPL_BASE
    tmp_size = _sort_tmp_size(small_seg_len, sort_run_len, sort_impl)
    sort_kernel[(num_seg,)](
        x_flat,
        sort_gm,
        N_COLS=n,
        SEG_LEN=small_seg_len,
        NUM_SEG=num_seg,
        NUM_SEG_PER_ROW=num_seg_per_row,
        ROW_WORDS=sort_row_words,
        SEGS_PER_CORE=1,
        NUM_CORES_C=num_seg,
        TMP_SIZE=tmp_size,
        SORT_TOPK=sort_run_len,
        PROP_SEG=prop_seg,
        SORT_IMPL=sort_impl,
        multibuffer=False,
    )

    y_vals, y_idx = _new_topk_outputs(x, m, k)
    static_merge4_unpack_kernel[(m,)](
        sort_gm,
        y_vals,
        y_idx,
        ROW_WORDS=sort_row_words,
        RUN_LEN=sort_run_len,
        NUM_RUNS_C=num_seg_per_row,
        K=k,
        multibuffer=False,
        enable_select_analysis=False,
    )
    return y_vals, y_idx


def _run_stream16_smallk_merge_tree(
    src_gm,
    y_vals,
    y_idx,
    *,
    batch: int,
    num_runs: int,
    run_len: int,
    row_words: int,
    k: int,
):
    out_segs = _out_segs(batch, num_runs, k, run_len)
    if num_runs <= MAX_WAYS:
        stream_cap = triton.next_power_of_2(4 * run_len * 2)
        unpack_cap = triton.next_power_of_2(k)
        smallk_stream_merge4_unpack_kernel[(batch,)](
            src_gm,
            y_vals,
            y_idx,
            ROW_WORDS=row_words,
            RUN_LEN=run_len,
            NUM_RUNS_C=num_runs,
            K=k,
            STREAM_CAP=stream_cap,
            UNPACK_CAP=unpack_cap,
            multibuffer=False,
            enable_select_analysis=False,
        )
        return

    if batch <= 24 and num_runs > STREAM16_SMALLK_GROUP_RUNS and out_segs > 1:
        local_row_words = out_segs * k * 2
        local_words = batch * 2 * local_row_words
        local_gm = torch.empty(
            (local_words,), device=y_vals.device, dtype=torch.float32
        )
        stream_cap = triton.next_power_of_2(4 * run_len * 2)
        smallk_core_local_stream_merge_kernel[(batch, out_segs)](
            src_gm,
            local_gm,
            NUM_SEG_PER_ROW=num_runs,
            SORT_ROW_WORDS=row_words,
            LOCAL_ROW_WORDS=local_row_words,
            RUN_LEN=run_len,
            K=k,
            NUM_CORES_C=out_segs,
            STREAM_CAP=stream_cap,
            multibuffer=False,
            enable_select_analysis=False,
        )
        final_stream_cap = triton.next_power_of_2(4 * k * 2)
        unpack_cap = triton.next_power_of_2(k)
        if out_segs <= MAX_WAYS:
            smallk_stream_merge4_unpack_kernel[(batch,)](
                local_gm,
                y_vals,
                y_idx,
                ROW_WORDS=local_row_words,
                RUN_LEN=k,
                NUM_RUNS_C=out_segs,
                K=k,
                STREAM_CAP=final_stream_cap,
                UNPACK_CAP=unpack_cap,
                multibuffer=False,
                enable_select_analysis=False,
            )
        else:
            smallk_stream_merge_unpack_kernel[(batch,)](
                local_gm,
                y_vals,
                y_idx,
                ROW_WORDS=local_row_words,
                RUN_LEN=k,
                NUM_RUNS_C=out_segs,
                K=k,
                STREAM_CAP=final_stream_cap,
                UNPACK_CAP=unpack_cap,
                multibuffer=False,
                enable_select_analysis=False,
            )
        return

    stream_cap = triton.next_power_of_2(4 * run_len * 2)
    unpack_cap = triton.next_power_of_2(k)
    smallk_stream_merge_unpack_kernel[(batch,)](
        src_gm,
        y_vals,
        y_idx,
        ROW_WORDS=row_words,
        RUN_LEN=run_len,
        NUM_RUNS_C=num_runs,
        K=k,
        STREAM_CAP=stream_cap,
        UNPACK_CAP=unpack_cap,
        multibuffer=False,
        enable_select_analysis=False,
    )


def topk(x: torch.Tensor, k: int, seg_len: int = 4096):
    if not HAS_TLE:
        raise RuntimeError(
            "Ascend topk is unavailable: requires triton.experimental.tle with "
            "the unified custom ops (tle.dsa.ascend.custom_ops)."
        )
    assert x.ndim == 2 and x.dtype == torch.float32
    m, n = x.shape
    num_seg_per_row = (n + seg_len - 1) // seg_len
    sort_run_len = k if k < seg_len else seg_len
    dispatch_sort_impl = _sort_impl(sort_run_len, seg_len)
    prefer_base_tiny = (
        m != 2
        and num_seg_per_row <= MAX_WAYS
        and dispatch_sort_impl == tle.dsa.ascend.custom_ops.SORT_IMPL_BASE
    )
    use_tiny_s_small_batch = _use_tiny_s_small_batch_path(m, n, k, seg_len)

    force_stream16_smallk = (
        _use_stream16_smallk_final(k, sort_run_len, num_seg_per_row)
        and not use_tiny_s_small_batch
    )

    if not force_stream16_smallk and use_tiny_s_small_batch:
        return _topk_tiny_s_small_batch_path(x, k)

    if (
        not force_stream16_smallk
        and _use_smalln_2048_path(m, n, seg_len)
        and (k > TINY_S_SMALL_BATCH_MAX_K or not prefer_base_tiny)
    ):
        return _topk_sort_then_final_merge(x, k, seg_len=2048)

    # Keep verified tiny-S paths: merge16 has too much fixed overhead for S<=4.
    if not force_stream16_smallk and prefer_base_tiny:
        return _topk_sort_then_final_merge(x, k, seg_len=seg_len)

    # Keep verified medium paths outside the configured batch-local target.
    if (
        not force_stream16_smallk
        and m != 2
        and num_seg_per_row > 16
        and _prefer_base_medium_path(m, num_seg_per_row, k, seg_len)
    ):
        return _topk_sort_then_final_merge(x, k, seg_len=seg_len)

    sort_gm, sort_row_words, sort_run_len, num_seg_per_row = (
        _build_sorted_runs_round_robin(x, k, seg_len=seg_len)
    )

    y_vals, y_idx = _new_topk_outputs(x, m, k)

    if force_stream16_smallk:
        _run_stream16_smallk_merge_tree(
            sort_gm,
            y_vals,
            y_idx,
            batch=m,
            num_runs=num_seg_per_row,
            run_len=sort_run_len,
            row_words=sort_row_words,
            k=k,
        )
        return y_vals, y_idx

    out_segs = _out_segs(m, num_seg_per_row, k, seg_len)
    if out_segs == num_seg_per_row:
        return _finish_with_final_merge_unpack(
            x,
            sort_gm,
            k=k,
            num_runs=num_seg_per_row,
            run_len=sort_run_len,
            row_words=sort_row_words,
            y_vals=y_vals,
            y_idx=y_idx,
        )

    input_segs_per_out_seg = _ceil_div(num_seg_per_row, out_segs)
    core_work_len = input_segs_per_out_seg * sort_run_len
    core_out_len = min(core_work_len, k)
    use_strided_final = _use_strided_local_final_merge(
        out_segs, num_seg_per_row, k, core_out_len, sort_run_len
    )
    use_single_run_tmp_unpack = _use_single_run_tmp_unpack(out_segs, k, core_out_len)

    tmp_row_words = out_segs * core_work_len * 2
    tmp_words = m * 2 * tmp_row_words
    tmp_gm = torch.empty((tmp_words,), device=x.device, dtype=torch.float32)

    final_row_words = out_segs * core_out_len * 2
    final_words = m * 2 * final_row_words
    final_gm = torch.empty((final_words,), device=x.device, dtype=torch.float32)

    local_chunk = CORE_LOCAL_CHUNK
    local_in_cap = MAX_WAYS * local_chunk * 2
    local_out_cap = MAX_WAYS * local_chunk * 2
    local_max_stream = (core_work_len // local_chunk) + 8

    core_local_kernel = core_local_nocopy_merge_kernel
    core_local_kwargs = {
        "LOCAL_GROUPS_C": _ceil_div(input_segs_per_out_seg, MAX_WAYS),
    }

    core_local_kernel[(m, out_segs)](
        sort_gm,
        tmp_gm,
        final_gm,
        NUM_SEG_PER_ROW=num_seg_per_row,
        SORT_RUN_LEN=sort_run_len,
        SORT_ROW_WORDS=sort_row_words,
        TMP_ROW_WORDS=tmp_row_words,
        FINAL_ROW_WORDS=final_row_words,
        CORE_WORK_LEN=core_work_len,
        CORE_OUT_LEN=core_out_len,
        K=k,
        CHUNK_C=local_chunk,
        IN_CAP=local_in_cap,
        OUT_CAP=local_out_cap,
        MAX_STREAM_ITERS=local_max_stream,
        MAX_WAYS_C=MAX_WAYS,
        NUM_CORES_C=out_segs,
        SKIP_FINAL_COPY=use_strided_final or use_single_run_tmp_unpack,
        **core_local_kwargs,
        multibuffer=False,
        enable_select_analysis=False,
    )

    # unpack_chunk feeds tl.arange(0, UNPACK_CHUNK) and must be a power of two.
    unpack_chunk = triton.next_power_of_2(min(k, CHUNK))
    num_chunks = (k + unpack_chunk - 1) // unpack_chunk

    if out_segs <= MAX_WAYS:
        if use_strided_final:
            final_src_phase = _final_phase(input_segs_per_out_seg)
            return _finish_with_final_merge_unpack(
                x,
                tmp_gm,
                k=k,
                num_runs=out_segs,
                run_len=core_out_len,
                row_words=tmp_row_words,
                run_stride=core_work_len,
                src_phase=final_src_phase,
                y_vals=y_vals,
                y_idx=y_idx,
            )

        if out_segs == 1:
            if use_single_run_tmp_unpack:
                final_src_phase = _final_phase(input_segs_per_out_seg)
                unpack_kernel[(m,)](
                    tmp_gm,
                    y_vals,
                    y_idx,
                    ROW_WORDS=tmp_row_words,
                    SRC_PHASE=final_src_phase,
                    K=k,
                    UNPACK_CHUNK=unpack_chunk,
                    NUM_CHUNKS=num_chunks,
                    multibuffer=False,
                    enable_select_analysis=False,
                )
                return y_vals, y_idx

        return _finish_with_final_merge_unpack(
            x,
            final_gm,
            k=k,
            num_runs=out_segs,
            run_len=core_out_len,
            row_words=final_row_words,
            y_vals=y_vals,
            y_idx=y_idx,
        )

    _launch_merge_unpack(
        final_gm,
        y_vals,
        y_idx,
        batch=m,
        num_runs=out_segs,
        run_len=core_out_len,
        row_words=final_row_words,
        k=k,
        chunk=CHUNK,
        unpack_chunk=unpack_chunk,
    )
    return y_vals, y_idx
