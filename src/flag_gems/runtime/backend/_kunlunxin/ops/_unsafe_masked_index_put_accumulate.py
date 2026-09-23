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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _dests_per_program(out_numel: int) -> int:
    """Number of output elements each program handles.

    Reusing one source-array read across DESTS targets cuts both launch count and
    memory traffic by DESTS. Cap at 32 to bound the `tl.static_range` unroll size.
    """
    if out_numel >= 512:
        return 32
    return max(1, min(32, triton.next_power_of_2(out_numel) // 16))


@libentry()
@triton.jit(do_not_specialize=["mask_numel", "out_numel"])
def _unsafe_masked_index_put_accumulate_kernel(
    out_ptr,
    inp_ptr,
    mask,
    index0,
    index1,
    index2,
    values,
    mask_numel,
    out_numel,
    SHAPE0: tl.constexpr,
    SHAPE1: tl.constexpr,
    SHAPE2: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    RANK: tl.constexpr,
    DESTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    dest_base = ext.program_id(0) * DESTS

    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < mask_numel
    keep = tl.load(mask + offsets, mask=active, other=0) != 0

    # Compute each source's flat destination offset. Clamp out-of-range/negative
    # indices before use: this backend hands the address to gm2lm before the mask
    # applies, so masking illegal indices still reads OOB and hangs the card.
    # Clamping to [0, SHAPE-1] matches torch's `index.clamp(-size, size-1)` for
    # non-negative indices.
    i0 = tl.load(index0 + offsets, mask=active, other=0).to(tl.int32)
    i0 = tl.minimum(tl.maximum(i0, 0), SHAPE0 - 1)
    dest = i0 * STRIDE0
    if RANK >= 2:
        i1 = tl.load(index1 + offsets, mask=active, other=0).to(tl.int32)
        i1 = tl.minimum(tl.maximum(i1, 0), SHAPE1 - 1)
        dest += i1 * STRIDE1
    if RANK >= 3:
        i2 = tl.load(index2 + offsets, mask=active, other=0).to(tl.int32)
        i2 = tl.minimum(tl.maximum(i2, 0), SHAPE2 - 1)
        dest += i2 * STRIDE2

    # Zero the update for masked-out sources and tail lanes, so the match below
    # needs no mask.
    update = tl.load(values + offsets, mask=active, other=0.0).to(tl.float32)
    update = tl.where(keep & active, update, 0.0)

    for c in tl.static_range(DESTS):
        out_off = dest_base + c
        # The out buffer keeps DESTS sentinel elements at the tail, so this store
        # needs no mask (a discrete store's mask is unreliable under address
        # collisions; prefer writing into the legal padding region).
        acc = tl.sum(tl.where(dest == out_off, update, 0.0), axis=0)
        in_off = tl.minimum(out_off, out_numel - 1)
        base = tl.load(inp_ptr + in_off).to(tl.float32)
        tl.store(out_ptr + out_off, base + acc)


# ---------------------------------------------------------------------------
# Multi-round "winner loop" path (large scale)
#
# The match kernel is O(out_numel * mask_numel), structurally unreachable at large
# scale, so we need O(mask_numel). On this backend both alternatives are blocked:
# atomic_add (~192 ns/elem, and pays full price even with an all-false mask) and
# sort/prefix-sum (multi-kernel, hits TritonXPU crash points). Hence an atomic-free
# winner loop: the natural single-winner semantics of a discrete store pick exactly
# one source per target each round; R rounds cover the maximum multiplicity.
#
# Key constraint: discrete-access cost depends only on how many lanes hit the same
# address (collisions serialize, ~192 ns/lane; random addresses ~1 ns/elem). So this
# design never lets two lanes share a sentinel slot:
#   * a source i, when retired or masked out, writes its own private slot
#     `out_numel + i`, not the shared out_numel;
#   * a target d with "no winner this round" is marked with its own value `d`
#     (translated to 0 via val_lookup), not the shared 0.
#
# tag address space (one row per round, row length row = out_numel + pad):
#   [0, out_numel)              target slots; value < out_numel means "no winner"
#   [out_numel, out_numel+pad)  source private slots; source i's marker = out_numel + i
# tag only needs an all-zero init (can't use torch.arange(int32): this backend
# ASSERT-FAILs and returns garbage). val_lookup is the same length as a tag row:
# [0, out_numel) is always 0, tail = values. combine rewrites the "no winner" index
# to `offs` (distinct per d, val_lookup[offs]==0), so one collision-free gather
# yields both "is there a winner" and "the winner's value".
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["mask_numel", "out_numel"])
def _umipa_prep_kernel(
    dest_buf,
    val_lookup,
    mask,
    index0,
    index1,
    index2,
    values,
    mask_numel,
    out_numel,
    SHAPE0: tl.constexpr,
    SHAPE1: tl.constexpr,
    SHAPE2: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flatten (index0..2, mask) into an int32 destination array and move values
    into the tail of val_lookup.

    Indices are clamped before use; masked-out sources write their private slot
    `out_numel + i` (so later round kernels never look at mask again); values go to
    `val_lookup[out_numel + i]`, so the marker is both the tag private-slot index and
    the value-lookup index.
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < mask_numel

    i0 = tl.load(index0 + offs, mask=inb, other=0).to(tl.int32)
    i0 = tl.minimum(tl.maximum(i0, 0), SHAPE0 - 1)
    dest = i0 * STRIDE0
    if RANK >= 2:
        i1 = tl.load(index1 + offs, mask=inb, other=0).to(tl.int32)
        i1 = tl.minimum(tl.maximum(i1, 0), SHAPE1 - 1)
        dest += i1 * STRIDE1
    if RANK >= 3:
        i2 = tl.load(index2 + offs, mask=inb, other=0).to(tl.int32)
        i2 = tl.minimum(tl.maximum(i2, 0), SHAPE2 - 1)
        dest += i2 * STRIDE2

    keep = tl.load(mask + offs, mask=inb, other=0) != 0
    marker = (out_numel + offs).to(tl.int32)
    tl.store(dest_buf + offs, tl.where(inb & keep, dest, marker))
    v = tl.load(values + offs, mask=inb, other=0.0)
    tl.store(val_lookup + out_numel + offs, v)


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_round_kernel(dest_buf, tag_prev, tag_cur, out_numel, BLOCK: tl.constexpr):
    """One round of the winner loop (fused variant).

    Whoever's marker survived in tag_prev[dest] is last round's winner: remove it
    from dest_buf (address becomes its private slot marker) and write `dest`
    (< out_numel) to tag_cur[dest] to mean "no winner yet this round"; still-alive
    lanes write their own marker to tag_cur[dest] to claim this round. The discrete
    store guarantees at most one marker survives per target.

    Known suboptimal: the retiring winner also writes tag_cur[dest], overwriting
    other alive lanes' claims on the same target => each multiplicity level burns
    ~two rounds. The fix is to let the winner write only its private slot, but
    `tl.store(tag_cur + tl.where(win, marker, d), marker)` reliably triggers 721
    (illegal address) and wedges the card on this backend, so large scale uses the
    retire+claim split arm instead (see `_ROUND_SPLIT_MIN_MASK_NUMEL`); small scale
    has enough rounds, so keep this fused variant.
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    w = tl.load(tag_prev + d)
    marker = (out_numel + offs).to(tl.int32)
    win = w == marker
    tl.store(dest_buf + offs, tl.where(win, marker, d))
    tl.store(tag_cur + d, tl.where(win, d, marker))


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_retire_kernel(dest_buf, tag_prev, out_numel, BLOCK: tl.constexpr):
    """Split arm, first half: only retire last round's winners, writing only the
    contiguous dest_buf and never touching tag.

    This avoids the fused variant's tl.where-computed store address (which hits 721).
    At r=0 tag_prev is an all-zero row, so `w == marker` is always false (no retire).
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    w = tl.load(tag_prev + d)
    marker = (out_numel + offs).to(tl.int32)
    tl.store(dest_buf + offs, tl.where(w == marker, marker, d))


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_claim_kernel(dest_buf, tag_cur, out_numel, BLOCK: tl.constexpr):
    """Split arm, second half: still-alive lanes claim this round.

    Here d is already the post-retire value: a retired source holds its private slot,
    so its store lands on tag_cur[out_numel+i], never touching any target and never
    overwriting another lane's claim. The store address is the loaded d, a known-good
    address form.
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    marker = (out_numel + offs).to(tl.int32)
    tl.store(tag_cur + d, marker)


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_finish_kernel(dest_buf, tag_last, alive, out_numel, BLOCK: tl.constexpr):
    """Collect the last round's winners (they'd only be removed by the next round,
    so do it here) and count remaining alive sources per program.

    `d < out_numel` means "still alive".
    """
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    w = tl.load(tag_last + d)
    marker = (out_numel + offs).to(tl.int32)
    d = tl.where(w == marker, marker, d)
    tl.store(dest_buf + offs, d)
    tl.store(alive + pid, tl.sum((d < out_numel).to(tl.int32), axis=0))


@libentry()
@triton.jit(do_not_specialize=["out_numel", "row"])
def _umipa_combine_kernel(
    out_ptr,
    src_ptr,
    tag,
    val_lookup,
    out_numel,
    row,
    ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-target reduction: out[d] = src[d] + sum_r val_lookup[tag[r][d]].

    `tag[r][d] >= out_numel` means round r has a winner on d; that value is the
    winner's marker and val_lookup[marker] is its value. Otherwise use `offs` as the
    index (val_lookup[offs]==0). "No winner" uses a distinct index per d to avoid
    address collisions. The tag gather is intentionally maskless: tail lanes read the
    private-slot region (legal and distinct addresses), and the result is dropped by
    the store mask.
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < out_numel
    acc = tl.load(src_ptr + offs, mask=inb, other=0.0).to(tl.float32)
    limit = out_numel.to(tl.int32)
    self_idx = offs.to(tl.int32)
    for r in tl.static_range(ROUNDS):
        tid = tl.load(tag + (r + 1) * row + offs)
        tid = tl.where(tid >= limit, tid, self_idx)
        acc += tl.load(val_lookup + tid).to(tl.float32)
    tl.store(out_ptr + offs, acc, mask=inb)


# Rounds per batch. The fused variant's winner overwrites other claims on the same
# target, so each multiplicity level burns ~two rounds; 8 covers typical inputs
# (Poisson multiplicity 4~9). If rounds run short, the convergence loop runs another
# batch automatically.
_ROUNDS_PER_BATCH = 8
# The four multi-round kernels use neither isCloseVectorization nor buffer_size_limit
# (measured to not affect this bottleneck, which is address collisions); kept as a
# constant only to make parametric A/B arm-switching a one-liner.
_ROUND_LAUNCH_KW = {}
# match path ~ out_numel*mask_numel*74ps; multi-round path has a ~11-launch floor;
# the two cross over around 4e6.
_MULTI_ROUND_MIN_WORK = 4_000_000
# Size threshold for splitting each round into retire+claim (split when mask_numel >=
# this). Splitting makes rounds == max multiplicity and halves discrete traffic, but
# costs `rounds` extra launches per batch; the crossover is mask_numel ~ 14000 => use
# 16384. Of the four benchmark shapes, only (2,1024,64) (mask=131072) triggers the
# split; (4096,) stays fused. Correctness does not depend on this value, only perf.
_ROUND_SPLIT_MIN_MASK_NUMEL = 16384


def _unsafe_masked_index_put_accumulate_multi_round(
    inp, mask_c, idx_c, values_c, rank, shape, strides
):
    out_numel = inp.numel()
    mask_numel = mask_c.numel()
    rounds = _ROUNDS_PER_BATCH
    round_split = mask_numel >= _ROUND_SPLIT_MIN_MASK_NUMEL

    block = max(64, min(2048, triton.next_power_of_2(mask_numel)))
    grid_m = (triton.cdiv(mask_numel, block),)
    m_pad = grid_m[0] * block
    block_n = max(64, min(2048, triton.next_power_of_2(out_numel)))
    grid_n = (triton.cdiv(out_numel, block_n),)

    # Row length row = out_numel target slots + one private slot per source; also
    # ensures the maskless tag gather in combine (offs up to grid_n*block_n-1) stays
    # in bounds.
    pad = max(m_pad, grid_n[0] * block_n - out_numel)
    row = out_numel + pad
    dev = inp.device

    # An all-zero row suffices (0 < out_numel always means "no winner" and never
    # equals any marker). Can't use torch.arange(int32): this backend ASSERT-FAILs
    # and returns garbage.
    tag = torch.zeros((rounds + 1) * row, dtype=torch.int32, device=dev)

    dest_buf = torch.empty(m_pad, dtype=torch.int32, device=dev)
    # The first out_numel entries must stay 0 (target contributes nothing this round);
    # prep fills the tail with values.
    val_lookup = torch.zeros(row, dtype=inp.dtype, device=dev)
    alive = torch.empty(grid_m[0], dtype=torch.int32, device=dev)
    out = torch.empty_like(inp)

    with torch_device_fn.device(dev):
        _umipa_prep_kernel[grid_m](
            dest_buf,
            val_lookup,
            mask_c,
            idx_c[0],
            idx_c[1],
            idx_c[2],
            values_c,
            mask_numel,
            out_numel,
            SHAPE0=shape[0],
            SHAPE1=shape[1],
            SHAPE2=shape[2],
            STRIDE0=strides[0],
            STRIDE1=strides[1],
            STRIDE2=strides[2],
            RANK=rank,
            BLOCK=block,
            **_ROUND_LAUNCH_KW,
        )
        src = inp
        for batch in range(64):
            if batch:
                tag.zero_()
            for r in range(rounds):
                if round_split:
                    _umipa_retire_kernel[grid_m](
                        dest_buf,
                        tag[r * row :],
                        out_numel,
                        BLOCK=block,
                        **_ROUND_LAUNCH_KW,
                    )
                    _umipa_claim_kernel[grid_m](
                        dest_buf,
                        tag[(r + 1) * row :],
                        out_numel,
                        BLOCK=block,
                        **_ROUND_LAUNCH_KW,
                    )
                else:
                    _umipa_round_kernel[grid_m](
                        dest_buf,
                        tag[r * row :],
                        tag[(r + 1) * row :],
                        out_numel,
                        BLOCK=block,
                        **_ROUND_LAUNCH_KW,
                    )
            _umipa_finish_kernel[grid_m](
                dest_buf,
                tag[rounds * row :],
                alive,
                out_numel,
                BLOCK=block,
                **_ROUND_LAUNCH_KW,
            )
            _umipa_combine_kernel[grid_n](
                out,
                src,
                tag,
                val_lookup,
                out_numel,
                row,
                ROUNDS=rounds,
                BLOCK=block_n,
                **_ROUND_LAUNCH_KW,
            )
            src = out
            # The only device->host sync: read alive once after a batch of rounds.
            # The reduction must run host-side: this backend's gems device sum (which
            # alive.sum() dispatches to under use_gems) illegally accesses memory
            # (error 700) and wedges the card on small tensors; .cpu() is just a D2H
            # copy followed by a CPU sum, avoiding the defect.
            if int(alive.cpu().sum()) == 0:
                break
        else:
            raise RuntimeError(
                "Kunlunxin _unsafe_masked_index_put_accumulate did not converge"
            )
    return out


def _unsafe_masked_index_put_accumulate(input, mask, indices, values):
    logger.debug("GEMS_KUNLUNXIN _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE")
    rank = input.ndim
    if rank < 1 or rank > 3 or len(indices) != rank:
        raise RuntimeError(
            "Kunlunxin _unsafe_masked_index_put_accumulate supports ranks 1 to 3"
        )
    # This aten op is functional (self is immutable; the reference clones then
    # index_put_), so it must return a new tensor.
    if input.numel() == 0 or mask.numel() == 0:
        return input.clone()

    inp = input if input.is_contiguous() else input.contiguous()
    mask_contiguous = mask.contiguous()
    values_contiguous = values.contiguous()
    contiguous_indices = [index.contiguous() for index in indices]
    while len(contiguous_indices) < 3:
        contiguous_indices.append(contiguous_indices[0])

    shape = list(inp.shape) + [1] * (3 - rank)
    strides = list(inp.stride()) + [0] * (3 - rank)
    out_numel = inp.numel()

    # Size split: small scale, the O(N*M) match needs only one launch; large scale,
    # match is structurally unreachable, so take the multi-round path.
    if out_numel * mask.numel() >= _MULTI_ROUND_MIN_WORK:
        return _unsafe_masked_index_put_accumulate_multi_round(
            inp, mask_contiguous, contiguous_indices, values_contiguous,
            rank, shape, strides,
        ).view(inp.shape)

    dests = _dests_per_program(out_numel)
    grid = (triton.cdiv(out_numel, dests),)
    # Tail sentinels: grid*dests may exceed out_numel; the extra lanes write into the
    # padding region instead of being masked out.
    out_buf = torch.empty(grid[0] * dests, dtype=inp.dtype, device=inp.device)
    out = out_buf[:out_numel].view(inp.shape)
    block_size = triton.next_power_of_2(mask.numel())

    with torch_device_fn.device(input.device):
        _unsafe_masked_index_put_accumulate_kernel[grid](
            out_buf,
            inp,
            mask_contiguous,
            contiguous_indices[0],
            contiguous_indices[1],
            contiguous_indices[2],
            values_contiguous,
            mask.numel(),
            out_numel,
            SHAPE0=shape[0],
            SHAPE1=shape[1],
            SHAPE2=shape[2],
            STRIDE0=strides[0],
            STRIDE1=strides[1],
            STRIDE2=strides[2],
            RANK=rank,
            DESTS=dests,
            BLOCK_SIZE=block_size,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return out
