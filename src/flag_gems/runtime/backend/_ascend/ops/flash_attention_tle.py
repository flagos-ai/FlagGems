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

"""FlashAttention forward for Ascend, written with the TLE tile DSA.

This is a narrow fast path, not a replacement for the `_attn_fwd` kernel in
`attention.py`: it handles fp16 self-attention with D == 64, no attention mask
and no dropout, and it does not produce the logsumexp that backward needs.
`can_use_tle_fa` decides; everything it rejects falls back to `_attn_fwd`.

Shapes: q[B, Hq, S, D], k/v[B, Hkv, S, D]. GQA/MQA works through
`Hq % Hkv == 0`; `is_causal=True` selects the causal variant.

The kernel is a cv-mixed 4-stage software pipeline. `grid = (num_cores,)` and
each program drives one Cube and one Vector engine, with the two scopes skewed
by one task so that on tick `g`:

    Cube    MM1(g) = Q @ K^T          MM2(g-1) = P @ V
    Vector  Vec1(g) = online softmax  Vec2(g-1) = rescale + accumulate

Stages hand off through GM ring buffers (`workspace_s`, `workspace_p` aliased
onto it, and `workspace_pv`) guarded by cross-core semaphores. The softmax
rescale factors need no buffer: Vec1 and Vec2 both run on the Vector core, so
program order plus a register snapshot is the whole protocol.

A task is one (output tile, KV group) pair, where a group is `CB` adjacent KV
blocks. Under causal the host both narrows the KV range per tile and balances
the resulting uneven task counts across cores.
"""

import contextlib
import json
import logging
import os
import traceback
from typing import NamedTuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle as tle  # noqa: F401  registers the tle dialects
    from triton.experimental.tle.language.dsa.ascend import (
        PIPE,
        sync_block_set,
        sync_block_wait,
    )

    HAS_TLE_DSA = True
except ImportError:  # pragma: no cover - depends on the installed triton
    tle = None
    PIPE = None
    sync_block_set = sync_block_wait = None
    HAS_TLE_DSA = False


# =============================================================================
#  Compile options
# =============================================================================
# Every cross-core dependency in this kernel is hand-written with
# sync_block_set/wait, so the backend's automatic sync injection and
# cv-workspace management are pure overhead and must be disabled.
#
# The flagtree Ascend backend reads USE_CUSTOM_COMPILE_OPT once, at import, into
# a module global, and then applies it to every linalg->binary compile in the
# process (it forces num_stages=1 and disables auto block-sync). Setting it at
# import here would therefore change codegen for every other triton kernel in
# the process. Patch the already-imported module global around the launch
# instead, which is what the compile actually reads.
_CUSTOM_COMPILE_FLAG = "_USE_CUSTOM_COMPILE_OPT"


@contextlib.contextmanager
def _custom_compile_opt():
    """Enable the backend's custom compile options for this launch only."""
    try:
        import triton.backends.ascend.compiler as backend_compiler
    except ImportError:
        yield
        return

    had = hasattr(backend_compiler, _CUSTOM_COMPILE_FLAG)
    previous = getattr(backend_compiler, _CUSTOM_COMPILE_FLAG, None)
    setattr(backend_compiler, _CUSTOM_COMPILE_FLAG, "1")
    try:
        yield
    finally:
        if had:
            setattr(backend_compiler, _CUSTOM_COMPILE_FLAG, previous)
        else:  # pragma: no cover - the attribute exists in supported backends
            delattr(backend_compiler, _CUSTOM_COMPILE_FLAG)


# =============================================================================
#  Configuration
# =============================================================================
DIM = 64  # the only head dim this kernel is written for

UB_BYTES = 192 * 1024  # per-Vector-core unified buffer
L0C_BYTES = 128 * 1024  # fp32 matmul accumulator budget
MAX_EVENT_ID = 16  # sync_block event_id lives in [0, 16)
MAX_RING = MAX_EVENT_ID // 2  # two semaphore banks are needed

# The Vec1 -> Vec2 hand-off is unrolled into registers named `rescale_<cb><sub>`,
# which caps both axes. Exceeding either would read an unwritten register and
# silently corrupt the output, so `config_is_valid` rejects it.
MAX_CB = 2
MAX_SUB_VEC_NUM = 2

BLOCK_M = 128
BLOCK_N = 256
RING = 2  # task-ring depth
# The Vector scope walks a BLOCK_M x BLOCK_N softmax tile as SUB_VEC_NUM
# sequential SUB_M x BLOCK_N sub-tiles, halving peak UB so that
# BLOCK_M=128/BLOCK_N=256 fits in 192 KB. The Cube side still works on full
# BLOCK_M tiles.
SUB_VEC_NUM = 2
SUB_M = BLOCK_M // SUB_VEC_NUM
NUM_STAGES = 4

DEFAULT_NUM_CORES = 20

# Cross-core semaphores. Hardware keys a channel by
# (event_id, sender_pipe, receiver_pipe), so the same id is reusable when the
# pipe pair differs. Per ring slot the id is `bank + slot`:
#
#   bank 0     S_READY  cube->vector  FIX,  MTE2
#              S_FREE   vector->cube  MTE2, FIX
#              P_READY  vector->cube  MTE3, MTE2
#              P_FREE   cube->vector  MTE2, MTE3
#   bank RING  PV_READY cube->vector  FIX,  MTE2
#              PV_FREE  vector->cube  MTE2, FIX
#
# PV needs its own bank because it reuses S's pipe pairs.

_num_cores_cache = {}


def num_cube_cores():
    """AI-core count, i.e. Cube/Vector engine pairs.

    Not `_ascend.utils.CORE_NUM`: that reports `num_vectorcore`, and this kernel
    launches one program per *pair*, driving both engines from the one program.
    """
    key = torch.npu.current_device() if torch.npu.is_available() else -1
    hit = _num_cores_cache.get(key)
    if hit is not None:
        return hit
    cores = DEFAULT_NUM_CORES
    try:
        import triton.runtime.driver as driver

        cores = int(driver.active.utils.get_device_properties(key)["num_aicore"])
    except (ImportError, AttributeError, RuntimeError, KeyError, ValueError):
        cores = DEFAULT_NUM_CORES
    if cores <= 0:
        cores = DEFAULT_NUM_CORES
    _num_cores_cache[key] = cores
    return cores


class FAConfig(NamedTuple):
    """One launch configuration.

    Every field reaches the kernel as a constexpr, except `num_stages` which is
    a launch attribute. The first four also decide host-side state:
    block_m/block_n/combine_batch/ring size the GM ring workspaces, and
    block_m/block_n/combine_batch determine the causal `task_map` contents.
    """

    block_m: int = BLOCK_M
    block_n: int = BLOCK_N
    combine_batch: int = 2
    sub_vec_num: int = SUB_VEC_NUM
    ring: int = RING
    num_stages: int = NUM_STAGES

    @property
    def sub_m(self):
        return self.block_m // self.sub_vec_num

    def label(self):
        return (
            f"M{self.block_m}/N{self.block_n}/CB{self.combine_batch}/"
            f"SV{self.sub_vec_num}/R{self.ring}/ns{self.num_stages}"
        )


DEFAULT_CONFIG = FAConfig()

# Coordinate-descent axes, each best-known-first so the baseline is candidate 0.
TUNE_SPACE = {
    "block_m": [128, 64, 256],
    "block_n": [256, 128],
    "combine_batch": [2, 1],
    "sub_vec_num": [2, 1],
    "ring": [2, 3, 4],
    "num_stages": [4, 2],
}


def _ub_bytes(cfg):
    """Peak Vector-core UB for one sub-tile pass, in bytes.

    Live at the widest point of `_vec1_softmax`: the score sub-tile, its
    exponentiated copy bound for `workspace_p`, and one fp32 temporary. Vec2's
    accumulator and PV load are charged too, since both stages share the core
    within a tick.
    """
    sub = cfg.sub_m * cfg.block_n
    vec1 = 2 * sub * 2 + sub * 4
    vec2 = 2 * (cfg.sub_m * DIM) * 2
    return vec1 + vec2


def _l0c_bytes(cfg):
    """Cube-side fp32 accumulator for one `tl.dot`; MM1 dominates MM2."""
    return cfg.block_m * cfg.block_n * 4


def _workspace_bytes(cfg, num_cores):
    """GM ring footprint across all cores. `workspace_p` aliases `workspace_s`."""
    per_core = cfg.ring * cfg.combine_batch * cfg.block_m * (cfg.block_n + DIM) * 2
    return num_cores * per_core


def config_is_valid(cfg, S):
    """Check a config against the kernel's invariants. Returns (ok, reason)."""
    if cfg.block_m <= 0 or cfg.block_n <= 0:
        return False, "non-positive block size"
    if S % cfg.block_m:
        return False, f"S={S} not divisible by BLOCK_M={cfg.block_m}"
    if S % cfg.block_n:
        return False, f"S={S} not divisible by BLOCK_N={cfg.block_n}"
    if cfg.sub_vec_num < 1 or cfg.sub_vec_num > MAX_SUB_VEC_NUM:
        return (
            False,
            f"SUB_VEC_NUM={cfg.sub_vec_num} outside [1,{MAX_SUB_VEC_NUM}] "
            "(register hand-off)",
        )
    if cfg.block_m % cfg.sub_vec_num:
        return (
            False,
            f"BLOCK_M={cfg.block_m} not divisible by SUB_VEC_NUM={cfg.sub_vec_num}",
        )
    if cfg.combine_batch < 1 or cfg.combine_batch > MAX_CB:
        return (
            False,
            f"CB={cfg.combine_batch} outside [1,{MAX_CB}] (register hand-off)",
        )
    nkv = S // cfg.block_n
    if nkv % cfg.combine_batch:
        return (
            False,
            f"num_kv_blocks={nkv} not divisible by CB={cfg.combine_batch}",
        )
    if cfg.ring < 2:
        return False, f"RING={cfg.ring} < 2 leaves no slot to overlap"
    if cfg.ring > MAX_RING:
        return (
            False,
            f"two banks * RING={cfg.ring} exceed event_id [0,{MAX_EVENT_ID})",
        )
    if cfg.num_stages < 1:
        return False, "num_stages < 1"
    if _ub_bytes(cfg) > UB_BYTES:
        return False, f"UB {_ub_bytes(cfg)} > {UB_BYTES}"
    if _l0c_bytes(cfg) > L0C_BYTES:
        return False, f"L0C {_l0c_bytes(cfg)} > {L0C_BYTES}"
    return True, "ok"


# =============================================================================
#  Task decode
# =============================================================================
@triton.jit
def _task_coords(
    t,
    tile_base,
    task_map,
    num_seq_blocks,
    heads_q,
    gqa_group,
    conbined_block_num,
    SPARSE: tl.constexpr,
):
    """Decode this core's flat task id `t` into tile coordinates and KV group.

    Dense (`SPARSE=False`): every tile owns `conbined_block_num` tasks, so the
    split is arithmetic and `tile_base` is the core's first tile.

    Causal (`SPARSE=True`): tiles own different task counts, so the host packs
    the whole mapping as `output_block_id * conbined_block_num + idx_in_conbine`
    and the kernel decodes with one scalar load plus a divmod.

    Returns (idx_in_conbine, global_head_idx, head_idx, batch_idx, kv_head_idx).
    """
    if SPARSE:
        packed = tl.load(task_map + t)
        output_block_id = packed // conbined_block_num
        idx_in_conbine = packed % conbined_block_num
    else:
        idx_in_conbine = t % conbined_block_num
        output_block_id = tile_base + t // conbined_block_num

    global_head_idx = output_block_id % num_seq_blocks  # Q row-block index
    head_idx = (output_block_id // num_seq_blocks) % heads_q
    batch_idx = output_block_id // (num_seq_blocks * heads_q)
    kv_head_idx = head_idx // gqa_group  # GQA: several Q heads share one KV head
    return idx_in_conbine, global_head_idx, head_idx, batch_idx, kv_head_idx


# =============================================================================
#  Cube stage 1 — MM1
# =============================================================================
@triton.jit
def _mm1_qkt(
    Q,
    K,
    workspace_s,
    cid,
    idx_in_conbine,
    global_head_idx,
    batch_idx,
    head_idx,
    kv_head_idx,
    ring_slot,
    sQb,
    sQh,
    sQs,
    sQd,
    sKb,
    sKh,
    sKs,
    sKd,
    S,
    CB: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    """MM1: S_part = Q @ K^T over CB blocks -> workspace_s[cid, ring_slot].

    Q is re-fetched per task: Cube register state does not survive a scope
    switch, since the two scopes compile to interleaved instruction streams that
    share L1.

    Each KV block gets a fresh zero accumulator, so a `tl.dot` result is that
    block's score directly, and it lands in GM before any Vector arithmetic
    touches it -- on this architecture L2/GM is the only Cube->Vector data path.
    """
    # workspace_s[ring_slot] free? (Vec1 released it RING ticks ago)
    sync_block_wait("vector", "cube", ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    q_row_offs = global_head_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_col_offs = tl.arange(0, DIM)
    q_ptr = (
        Q
        + batch_idx * sQb
        + head_idx * sQh
        + q_row_offs[:, None] * sQs
        + q_col_offs[None, :] * sQd
    )
    q_l1 = tl.load(q_ptr)

    for cb_idx in tl.static_range(CB):
        kv_idx = idx_in_conbine * CB + cb_idx

        k_bp = tl.make_block_ptr(
            K + batch_idx * sKb + kv_head_idx * sKh,
            (S, DIM),
            (sKs, sKd),
            (kv_idx * BLOCK_N, 0),
            (BLOCK_N, DIM),
            (1, 0),
        )
        k_block = tl.load(k_bp)

        # Both operands stay plain register tensors: routing either through
        # tile_to_tensor makes the lowered call see a 4D cbuf memref, which the
        # matmul intrinsic rejects (it wants 2D).
        attn_score_l0c = tl.dot(q_l1.to(tl.float16), tl.trans(k_block).to(tl.float16))

        score_store_bp = tl.make_block_ptr(
            workspace_s
            + (
                cid * RING * CB * BLOCK_M * BLOCK_N
                + ring_slot * CB * BLOCK_M * BLOCK_N
                + cb_idx * BLOCK_M * BLOCK_N
            ),
            (BLOCK_M, BLOCK_N),
            (BLOCK_N, 1),
            (0, 0),
            (BLOCK_M, BLOCK_N),
            (1, 0),
        )
        tl.store(score_store_bp, attn_score_l0c.to(workspace_s.dtype.element_ty))

    # every CB score block written -> Vec1 may start on this slot
    sync_block_set("cube", "vector", ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


# =============================================================================
#  Cube stage 3 — MM2
# =============================================================================
@triton.jit
def _mm2_pv(
    V,
    workspace_p,
    workspace_pv,
    cid,
    idx_in_conbine,
    batch_idx,
    kv_head_idx,
    ring_slot,
    sKb,
    sKh,
    sKs,
    sKd,
    S,
    CB: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    """MM2: O_part = P @ V over CB blocks -> workspace_pv[cid, ring_slot]."""
    # P from Vec1 ready?
    sync_block_wait("vector", "cube", ring_slot, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
    # workspace_pv[ring_slot] free? (Vec2 released it RING ticks ago)
    sync_block_wait("vector", "cube", RING + ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    for cb_idx in tl.static_range(CB):
        kv_idx = idx_in_conbine * CB + cb_idx

        v_bp = tl.make_block_ptr(
            V + batch_idx * sKb + kv_head_idx * sKh,
            (S, DIM),
            (sKs, sKd),
            (kv_idx * BLOCK_N, 0),
            (BLOCK_N, DIM),
            (1, 0),
        )
        v_block = tl.load(v_bp)

        prob_load_bp = tl.make_block_ptr(
            workspace_p
            + (
                cid * RING * CB * BLOCK_M * BLOCK_N
                + ring_slot * CB * BLOCK_M * BLOCK_N
                + cb_idx * BLOCK_M * BLOCK_N
            ),
            (BLOCK_M, BLOCK_N),
            (BLOCK_N, 1),
            (0, 0),
            (BLOCK_M, BLOCK_N),
            (1, 0),
        )
        p_block = tl.load(prob_load_bp)

        pv_part_l0c = tl.dot(p_block.to(tl.float16), v_block.to(tl.float16))

        pv_store_bp = tl.make_block_ptr(
            workspace_pv
            + cid * RING * CB * BLOCK_M * DIM
            + ring_slot * CB * BLOCK_M * DIM
            + cb_idx * BLOCK_M * DIM,
            (BLOCK_M, DIM),
            (DIM, 1),
            (0, 0),
            (BLOCK_M, DIM),
            (1, 0),
        )
        tl.store(pv_store_bp, pv_part_l0c.to(workspace_pv.dtype.element_ty))

    # P consumed -> Vec1 may reuse the slot;  PV ready -> Vec2 may start
    sync_block_set("cube", "vector", ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)
    sync_block_set("cube", "vector", RING + ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


@triton.jit
def _fp16_max_combine(a, b):
    """Element-wise max that stays in fp16.

    `tl.max` promotes fp16 to fp32 before reducing and the backend hoists that
    `extf` above the reduction, so every score block would pay a widen+narrow
    pair. An explicit combiner keeps the whole reduction in fp16.
    """
    return tl.maximum(a, b)


# =============================================================================
#  Vector stage 2 — Vec1 (online softmax)
# =============================================================================
@triton.jit
def _vec1_softmax(
    workspace_s,
    workspace_p,
    mask_bias,
    cid,
    idx_in_conbine,
    global_head_idx,
    ring_slot,
    running_max_0,
    running_max_1,
    IS_CAUSAL: tl.constexpr,
    CB: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUB_M: tl.constexpr,
    SUB_VEC_NUM: tl.constexpr,
    N_PAT: tl.constexpr,
):
    """Vec1: online softmax over CB score blocks -> workspace_p + registers.

    Reads workspace_s[ring_slot], applies the causal mask, stabilises and
    exponentiates, stores P for MM2, and returns the per-(KV block, sub-tile)
    rescale/expsum pair in registers.

    Returns (running_max_0, running_max_1,
             rescale_00, expsum_00, rescale_01, expsum_01,
             rescale_10, expsum_10, rescale_11, expsum_11).
    """
    # all CB score blocks ready?
    sync_block_wait("cube", "vector", ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
    # workspace_p[ring_slot] free? (MM2 released it)
    sync_block_wait("cube", "vector", ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)

    # Reset at each tile's first KV group. `running_max` holds the *negated*
    # max; 10.0 is a sentinel the select below displaces on the first block.
    if idx_in_conbine == 0:
        running_max_0 = tl.full((SUB_M,), 10.0, tl.float16)
        running_max_1 = tl.full((SUB_M,), 10.0, tl.float16)

    rescale_00 = tl.zeros((SUB_M,), tl.float16)
    expsum_00 = tl.zeros((SUB_M,), tl.float16)
    rescale_01 = tl.zeros((SUB_M,), tl.float16)
    expsum_01 = tl.zeros((SUB_M,), tl.float16)
    rescale_10 = tl.zeros((SUB_M,), tl.float16)
    expsum_10 = tl.zeros((SUB_M,), tl.float16)
    rescale_11 = tl.zeros((SUB_M,), tl.float16)
    expsum_11 = tl.zeros((SUB_M,), tl.float16)

    for cb_idx in tl.static_range(CB):
        kv_idx = idx_in_conbine * CB + cb_idx
        kv_col_start = kv_idx * BLOCK_N

        for sub_idx in tl.static_range(SUB_VEC_NUM):
            offset_m = sub_idx * SUB_M
            q_row_start = global_head_idx * BLOCK_M + offset_m

            score_load_bp = tl.make_block_ptr(
                workspace_s
                + cid * RING * CB * BLOCK_M * BLOCK_N
                + ring_slot * CB * BLOCK_M * BLOCK_N
                + cb_idx * BLOCK_M * BLOCK_N
                + offset_m * BLOCK_N,
                (SUB_M, BLOCK_N),
                (BLOCK_N, 1),
                (0, 0),
                (SUB_M, BLOCK_N),
                (1, 0),
            )
            attn_score_block = tl.load(score_load_bp)

            if sub_idx == 0:
                running_max = running_max_0
            else:
                running_max = running_max_1

            if IS_CAUSAL:
                # Add a precomputed bias plane instead of rebuilding the mask on
                # the vector unit: the bias is 0 where the score is kept and
                # -256 where it is masked, so `+` reproduces a `tl.where`.
                # Computing the mask instead costs an int32 compare chain
                # (`>=` lowers to vmax followed by vcmp_eq) several times the
                # score block's volume.
                #
                # `pat` is scalar arithmetic. d = q_row_start - kv_col_start is
                # the diagonal offset and only SUB_M multiples occur; the set is
                # the same for every S because d depends only on
                # BLOCK_M/BLOCK_N/CB. The clamp sends out-of-range groups to the
                # all-mask (0) or all-keep (N_PAT-1) plane, so no task needs a
                # runtime test.
                d = q_row_start - kv_col_start
                pat = (d + SUB_M) // SUB_M
                pat = max(min(pat, N_PAT - 1), 0)
                bias_bp = tl.make_block_ptr(
                    mask_bias + pat * SUB_M * BLOCK_N,
                    (SUB_M, BLOCK_N),
                    (BLOCK_N, 1),
                    (0, 0),
                    (SUB_M, BLOCK_N),
                    (1, 0),
                )
                attn_score_block = attn_score_block + tl.load(bias_bp)

            block_row_max = tl.reduce(
                attn_score_block, axis=-1, combine_fn=_fp16_max_combine
            )
            prev_neg_max = running_max

            # `tl.minimum` would lower to a NaN-correct expansion (two isnan
            # tests plus two selects) around 2 ops of real work. Neither operand
            # can be NaN here -- block_row_max is a max-reduce over finite
            # scores and running_max is either the sentinel or a previous result
            # of this expression -- so ask for the select directly.
            neg_block_row_max = -block_row_max
            candidate_neg_max = tl.where(
                neg_block_row_max < running_max, neg_block_row_max, running_max
            )

            if IS_CAUSAL:
                # A fully masked row has max == -256, which would drive exp()
                # and the denominator to a degenerate 0/0; hold the previous max
                # instead. Still needed under the narrowed schedule: a tile's
                # last CB group can overshoot its diagonal block, so the tail
                # blocks inside that group really are fully masked.
                #
                # The literal is pinned to fp16 because a bare Python float is
                # an fp32 constant and would promote the whole compare, while
                # every other term in this chain is already fp16.
                neg_200 = tl.full((1,), -200.0, tl.float16)
                row_all_masked = block_row_max < neg_200
                neg_max_new = tl.where(
                    row_all_masked, running_max, candidate_neg_max
                ).to(tl.float16)
            else:
                neg_max_new = candidate_neg_max.to(tl.float16)

            softmax_p = tl.exp(attn_score_block + neg_max_new[:, None])

            rescale = tl.exp(neg_max_new - prev_neg_max).to(tl.float16)
            block_expsum = tl.sum(softmax_p, axis=-1, keep_dims=False).to(tl.float16)

            # Register hand-off to Vec2, positional as <cb_idx><sub_idx>.
            if cb_idx == 0 and sub_idx == 0:
                rescale_00 = rescale
                expsum_00 = block_expsum
            if cb_idx == 0 and sub_idx == 1:
                rescale_01 = rescale
                expsum_01 = block_expsum
            if cb_idx == 1 and sub_idx == 0:
                rescale_10 = rescale
                expsum_10 = block_expsum
            if cb_idx == 1 and sub_idx == 1:
                rescale_11 = rescale
                expsum_11 = block_expsum

            prob_store_bp = tl.make_block_ptr(
                workspace_p
                + cid * RING * CB * BLOCK_M * BLOCK_N
                + ring_slot * CB * BLOCK_M * BLOCK_N
                + cb_idx * BLOCK_M * BLOCK_N
                + offset_m * BLOCK_N,
                (SUB_M, BLOCK_N),
                (BLOCK_N, 1),
                (0, 0),
                (SUB_M, BLOCK_N),
                (1, 0),
            )
            tl.store(prob_store_bp, softmax_p.to(workspace_p.dtype.element_ty))

            if sub_idx == 0:
                running_max_0 = neg_max_new
            else:
                running_max_1 = neg_max_new

    # scores consumed -> MM1 may reuse the slot;  P ready -> MM2 may start
    sync_block_set("vector", "cube", ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
    sync_block_set("vector", "cube", ring_slot, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)

    return (
        running_max_0,
        running_max_1,
        rescale_00,
        expsum_00,
        rescale_01,
        expsum_01,
        rescale_10,
        expsum_10,
        rescale_11,
        expsum_11,
    )


# =============================================================================
#  Vector stage 4 — Vec2 (FlashAttention online refresh)
# =============================================================================
@triton.jit
def _vec2_accumulate(
    Out,
    workspace_pv,
    cid,
    idx_in_conbine,
    global_head_idx,
    batch_idx,
    head_idx,
    ring_slot,
    acc_o_0,
    softmax_denom_0,
    acc_o_1,
    softmax_denom_1,
    prev_rescale_00,
    prev_expsum_00,
    prev_rescale_01,
    prev_expsum_01,
    prev_rescale_10,
    prev_expsum_10,
    prev_rescale_11,
    prev_expsum_11,
    last_kv_idx,
    sOb,
    sOh,
    sOs,
    sOd,
    S,
    CB: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
    SUB_M: tl.constexpr,
    SUB_VEC_NUM: tl.constexpr,
):
    """Vec2: acc_o = acc_o * rescale + P@V, finalising on the tile's last block.

    `last_kv_idx` is the tile's own final KV-block index rather than
    `num_kv_blocks - 1`, because under causal each tile stops at its diagonal.

    Returns (acc_o_0, softmax_denom_0, acc_o_1, softmax_denom_1).
    """
    # all CB P@V blocks ready?
    sync_block_wait("cube", "vector", RING + ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)

    for cb_idx in tl.static_range(CB):
        kv_idx = idx_in_conbine * CB + cb_idx

        for sub_idx in tl.static_range(SUB_VEC_NUM):
            offset_m = sub_idx * SUB_M

            pv_load_bp = tl.make_block_ptr(
                workspace_pv
                + cid * RING * CB * BLOCK_M * DIM
                + ring_slot * CB * BLOCK_M * DIM
                + cb_idx * BLOCK_M * DIM
                + offset_m * DIM,
                (SUB_M, DIM),
                (DIM, 1),
                (0, 0),
                (SUB_M, DIM),
                (1, 0),
            )
            pv_acc = tl.load(pv_load_bp)

            if cb_idx == 0 and sub_idx == 0:
                rescale = prev_rescale_00
                block_expsum = prev_expsum_00
            if cb_idx == 0 and sub_idx == 1:
                rescale = prev_rescale_01
                block_expsum = prev_expsum_01
            if cb_idx == 1 and sub_idx == 0:
                rescale = prev_rescale_10
                block_expsum = prev_expsum_10
            if cb_idx == 1 and sub_idx == 1:
                rescale = prev_rescale_11
                block_expsum = prev_expsum_11

            if sub_idx == 0:
                acc_o = acc_o_0
                softmax_denom = softmax_denom_0
            else:
                acc_o = acc_o_1
                softmax_denom = softmax_denom_1

            if kv_idx == 0:
                # The first KV block needs no rescale, just a seed.
                acc_o = pv_acc
                softmax_denom = block_expsum
            else:
                acc_o = acc_o * rescale[:, None] + pv_acc
                softmax_denom = softmax_denom * rescale + block_expsum

            if kv_idx == last_kv_idx:
                # Normalise and store straight into the final layout, with no
                # staging tile and no transpose.
                #
                # Reciprocal first, then multiply: `acc_o / softmax_denom[:, None]`
                # broadcasts the denominator into a full fp32 tile and divides
                # every element, where one division per row is all the row needs.
                denom_recip = 1.0 / softmax_denom
                output_block = (acc_o * denom_recip[:, None]).to(Out.dtype.element_ty)
                o_bp = tl.make_block_ptr(
                    Out + batch_idx * sOb + head_idx * sOh,
                    (S, DIM),
                    (sOs, sOd),
                    (global_head_idx * BLOCK_M + offset_m, 0),
                    (SUB_M, DIM),
                    (1, 0),
                )
                tl.store(o_bp, output_block)

            if sub_idx == 0:
                acc_o_0 = acc_o
                softmax_denom_0 = softmax_denom
            else:
                acc_o_1 = acc_o
                softmax_denom_1 = softmax_denom

    # PV consumed -> MM2 may reuse the slot
    sync_block_set("vector", "cube", RING + ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    return acc_o_0, softmax_denom_0, acc_o_1, softmax_denom_1


# =============================================================================
#  The skew-1 4-stage pipeline kernel
#
#  Semaphore ledger:
#
#      channel    producer     consumer
#      S_READY    MM1(g)       Vec1(g)
#      S_FREE     Vec1(g)      MM1(g+RING)
#      P_READY    Vec1(g)      MM2(g)
#      P_FREE     MM2(g)       Vec1(g+RING)
#      PV_READY   MM2(g)       Vec2(g)
#      PV_FREE    Vec2(g)      MM2(g+RING)
#
#  The softmax factors need no channel: producer and consumer are both on the
#  Vector core, so the snapshot into `prev_*` plus program order is the whole
#  protocol. Vec1(g) runs before Vec2(g-1) in the same tick and would otherwise
#  clobber the registers Vec2 still needs.
# =============================================================================
@triton.jit
def _flash_attention_fwd_kernel(
    Q,
    K,
    V,
    Out,
    workspace_s,
    workspace_p,
    workspace_pv,  # GM ring workspaces
    mask_bias,  # f16[N_PAT, SUB_M, BLOCK_N]: additive causal bias, 0 or -256
    task_map,  # i32[num_cores, max_tasks]: packed tile*cbn + group (causal only)
    core_ntasks,  # i32[num_cores]: tasks owned (causal only)
    max_tasks,  # row stride of task_map (causal only)
    B,
    Hq,
    Hkv,
    S,
    sQb,
    sQh,
    sQs,
    sQd,
    sKb,
    sKh,
    sKs,
    sKd,
    sOb,
    sOh,
    sOs,
    sOd,
    num_seq_blocks,
    heads_q,
    gqa_group,
    num_kv_blocks,  # KV blocks per output tile (= S // BLOCK_N)
    conbined_block_num,  # tasks per output tile (= num_kv_blocks // CB)
    block_num_per_core,
    rem_block_num,
    CB: tl.constexpr,  # KV blocks per task
    NUM_KV_BLOCKS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SPARSE: tl.constexpr,  # use the host's narrowed + balanced causal schedule
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
    SUB_M: tl.constexpr,
    SUB_VEC_NUM: tl.constexpr,
    N_PAT: tl.constexpr,
):
    cid = tl.program_id(0)

    # ---- task distribution ----
    if SPARSE:
        my_task_map = task_map + cid * max_tasks
        num_global_tasks = tl.load(core_ntasks + cid)
        tile_base = 0  # unused on the sparse path
    else:
        my_task_map = task_map
        block_start = cid * block_num_per_core + tl.where(
            cid < rem_block_num, cid, rem_block_num
        )
        block_num = block_num_per_core + tl.where(cid < rem_block_num, 1, 0)
        num_global_tasks = block_num * conbined_block_num
        tile_base = block_start

    # ---- Vector-side persistent state (UB registers) ----
    # One set per sub-tile: the sub-tiles run sequentially but each carries its
    # own FlashAttention accumulator across the tile's KV groups.
    acc_o_0 = tl.zeros((SUB_M, DIM), tl.float16)
    acc_o_1 = tl.zeros((SUB_M, DIM), tl.float16)
    softmax_denom_0 = tl.zeros((SUB_M,), tl.float16)
    softmax_denom_1 = tl.zeros((SUB_M,), tl.float16)
    running_max_0 = tl.full((SUB_M,), 10.0, tl.float16)
    running_max_1 = tl.full((SUB_M,), 10.0, tl.float16)

    # Vec1(g) -> Vec2(g) softmax factors, one pair per (cb_idx, sub_idx).
    rescale_00 = tl.zeros((SUB_M,), tl.float16)
    expsum_00 = tl.zeros((SUB_M,), tl.float16)
    rescale_01 = tl.zeros((SUB_M,), tl.float16)
    expsum_01 = tl.zeros((SUB_M,), tl.float16)
    rescale_10 = tl.zeros((SUB_M,), tl.float16)
    expsum_10 = tl.zeros((SUB_M,), tl.float16)
    rescale_11 = tl.zeros((SUB_M,), tl.float16)
    expsum_11 = tl.zeros((SUB_M,), tl.float16)

    # ---- prologue: prime the FREE channels so the first RING ticks proceed ----
    with tle.scope(core_mode="cube"):
        for s in tl.static_range(RING):
            sync_block_set("cube", "vector", s, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)

    with tle.scope(core_mode="vector"):
        for s in tl.static_range(RING):
            sync_block_set("vector", "cube", s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
            sync_block_set("vector", "cube", RING + s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    # ---- main loop; one trailing tick drains stage g-1 ----
    # Task g-1's coordinates ride in scalar registers rather than being decoded
    # twice: on the sparse path each decode is a GM load.
    p_idx_in_conbine = 0
    p_global_head_idx = 0
    p_head_idx = 0
    p_batch_idx = 0
    p_kv_head_idx = 0

    for g in range(num_global_tasks + 1):

        # One decode per tick, shared by both scopes.
        if g < num_global_tasks:
            (
                idx_in_conbine,
                global_head_idx,
                head_idx,
                batch_idx,
                kv_head_idx,
            ) = _task_coords(
                g,
                tile_base,
                my_task_map,
                num_seq_blocks,
                heads_q,
                gqa_group,
                conbined_block_num,
                SPARSE,
            )
        else:
            idx_in_conbine = 0
            global_head_idx = 0
            head_idx = 0
            batch_idx = 0
            kv_head_idx = 0

        # -----------------------------------------------------------------
        #  Cube scope: MM1(g), MM2(g-1)
        # -----------------------------------------------------------------
        with tle.scope(core_mode="cube"):
            if g < num_global_tasks:
                _mm1_qkt(
                    Q,
                    K,
                    workspace_s,
                    cid,
                    idx_in_conbine,
                    global_head_idx,
                    batch_idx,
                    head_idx,
                    kv_head_idx,
                    g % RING,
                    sQb,
                    sQh,
                    sQs,
                    sQd,
                    sKb,
                    sKh,
                    sKs,
                    sKd,
                    S,
                    CB,
                    RING,
                    BLOCK_M,
                    BLOCK_N,
                    DIM,
                )

            if g >= 1:
                _mm2_pv(
                    V,
                    workspace_p,
                    workspace_pv,
                    cid,
                    p_idx_in_conbine,
                    p_batch_idx,
                    p_kv_head_idx,
                    (g - 1) % RING,
                    sKb,
                    sKh,
                    sKs,
                    sKd,
                    S,
                    CB,
                    RING,
                    BLOCK_M,
                    BLOCK_N,
                    DIM,
                )

        # -----------------------------------------------------------------
        #  Vector scope: Vec1(g), Vec2(g-1)
        # -----------------------------------------------------------------
        with tle.scope(core_mode="vector"):
            # Vec2 below consumes tick g-1's factors, but Vec1(g) runs first and
            # overwrites the registers, so snapshot them here.
            prev_rescale_00 = rescale_00
            prev_expsum_00 = expsum_00
            prev_rescale_01 = rescale_01
            prev_expsum_01 = expsum_01
            prev_rescale_10 = rescale_10
            prev_expsum_10 = expsum_10
            prev_rescale_11 = rescale_11
            prev_expsum_11 = expsum_11

            if g < num_global_tasks:
                (
                    running_max_0,
                    running_max_1,
                    rescale_00,
                    expsum_00,
                    rescale_01,
                    expsum_01,
                    rescale_10,
                    expsum_10,
                    rescale_11,
                    expsum_11,
                ) = _vec1_softmax(
                    workspace_s,
                    workspace_p,
                    mask_bias,
                    cid,
                    idx_in_conbine,
                    global_head_idx,
                    g % RING,
                    running_max_0,
                    running_max_1,
                    IS_CAUSAL,
                    CB,
                    RING,
                    BLOCK_M,
                    BLOCK_N,
                    SUB_M,
                    SUB_VEC_NUM,
                    N_PAT,
                )

            if g >= 1:
                global_head_idx_1 = p_global_head_idx
                # Which KV block finalises this tile's output row?
                #
                # Causal: the tile stops at its diagonal block, rounded up to a
                # whole CB group -- the same closed form the host uses to size
                # the schedule, so no lookup table is needed.
                # Dense: every tile walks the full range.
                if SPARSE:
                    diag_kv = ((global_head_idx_1 + 1) * BLOCK_M - 1) // BLOCK_N
                    n_groups = (diag_kv + CB) // CB  # ceil((diag_kv + 1) / CB)
                    last_kv_idx = n_groups * CB - 1
                else:
                    last_kv_idx = NUM_KV_BLOCKS - 1

                (
                    acc_o_0,
                    softmax_denom_0,
                    acc_o_1,
                    softmax_denom_1,
                ) = _vec2_accumulate(
                    Out,
                    workspace_pv,
                    cid,
                    p_idx_in_conbine,
                    global_head_idx_1,
                    p_batch_idx,
                    p_head_idx,
                    (g - 1) % RING,
                    acc_o_0,
                    softmax_denom_0,
                    acc_o_1,
                    softmax_denom_1,
                    prev_rescale_00,
                    prev_expsum_00,
                    prev_rescale_01,
                    prev_expsum_01,
                    prev_rescale_10,
                    prev_expsum_10,
                    prev_rescale_11,
                    prev_expsum_11,
                    last_kv_idx,
                    sOb,
                    sOh,
                    sOs,
                    sOd,
                    S,
                    CB,
                    RING,
                    BLOCK_M,
                    DIM,
                    SUB_M,
                    SUB_VEC_NUM,
                )

        # Hand this tick's coordinates to next tick's g-1 stages.
        p_idx_in_conbine = idx_in_conbine
        p_global_head_idx = global_head_idx
        p_head_idx = head_idx
        p_batch_idx = batch_idx
        p_kv_head_idx = kv_head_idx

    # ---- epilogue: drain the primed FREE tokens no stage consumed ----
    with tle.scope(core_mode="cube"):
        for s in tl.static_range(RING):
            sync_block_wait("cube", "vector", s, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)

    with tle.scope(core_mode="vector"):
        for s in tl.static_range(RING):
            sync_block_wait("vector", "cube", s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
            sync_block_wait("vector", "cube", RING + s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)


# =============================================================================
#  Host: causal schedule and mask table
# =============================================================================
_schedule_cache = {}
_mask_bias_cache = {}
_workspace_cache = {}
_autotune_cache = {}


def _build_causal_schedule(
    num_seq_blocks,
    heads_q,
    batch,
    CB,
    num_kv_blocks,
    device,
    num_cores,
    block_m=BLOCK_M,
    block_n=BLOCK_N,
):
    """Build the narrowed, balanced schedule for a causal run.

    * narrow: Q row-block `i` spans rows `[i*BLOCK_M, (i+1)*BLOCK_M)`, so its
      last visible KV block is `((i+1)*BLOCK_M - 1) // BLOCK_N`. Tasks past that
      are never issued. Rounded up to whole CB groups, a tile owns
      `ceil((last_kv + 1) / CB)` tasks.
    * balance: tile costs now range from 1 task to `num_kv_blocks/CB`, so hand
      tiles out longest-first to whichever core is least loaded.

    Returns (task_map, core_ntasks, max_tasks): an i32[num_cores, max_tasks]
    device tensor of packed `tile * conbined_block_num + group` entries, the
    per-core task count, and the row stride.
    """
    row_block = torch.arange(num_seq_blocks, dtype=torch.int64)
    last_kv_per_row = ((row_block + 1) * block_m - 1) // block_n
    last_kv_per_row = torch.clamp(last_kv_per_row, max=num_kv_blocks - 1)
    # A task covers CB KV blocks, so round the count up to whole groups. The
    # final group may reach past `last_kv`; those blocks are fully masked and
    # Vec1's all-masked-row path keeps them out of the accumulator.
    tasks_per_row = (last_kv_per_row + 1 + CB - 1) // CB
    tile_tasks = tasks_per_row.repeat(batch * heads_q)

    # Longest-processing-time-first greedy packing.
    order = torch.argsort(tile_tasks, descending=True, stable=True)
    per_core_tiles = [[] for _ in range(num_cores)]
    per_core_load = [0] * num_cores
    for tile in order.tolist():
        c = min(range(num_cores), key=lambda i: per_core_load[i])
        per_core_tiles[c].append(tile)
        per_core_load[c] += int(tile_tasks[tile])

    # One packed i32 per task, so the kernel decodes with a single load. Cores
    # walk their tiles in ascending order, because consecutive tiles share a
    # (batch, head) and that keeps the Q/K/V reads local.
    conbined_block_num = num_kv_blocks // CB
    max_tasks = int(max(per_core_load))
    task_map = torch.zeros((num_cores, max_tasks), dtype=torch.int32)
    core_ntasks = torch.zeros((num_cores,), dtype=torch.int32)
    for c in range(num_cores):
        j = 0
        for tile in sorted(per_core_tiles[c]):
            for grp in range(int(tile_tasks[tile])):
                task_map[c, j] = tile * conbined_block_num + grp
                j += 1
        core_ntasks[c] = j

    return task_map.to(device), core_ntasks.to(device), max_tasks


def _dense_mask_bias_stub(device):
    """1-element stand-in for `mask_bias` on the dense path.

    The table is read only under `if IS_CAUSAL`, and allocating a real one
    anyway is not free: an unread buffer of that size shifts the ring
    workspaces around and measurably slowed several dense shapes.
    """
    key = str(device)
    hit = _mask_bias_cache.get(("stub", key))
    if hit is None:
        hit = torch.zeros((1,), dtype=torch.float16, device=device)
        _mask_bias_cache[("stub", key)] = hit
    return hit


def _build_mask_bias(sub_m, block_n, device):
    """Precomputed additive causal bias, cached per (shape, device).

    Plane `pat` covers diagonal offset `d = (pat - 1) * sub_m`, where
    `d = q_row_start - kv_col_start`. Within a plane, element (i, j) is 0 when
    the score is kept (`i + d >= j`) and -256 when masked.

    Only `d` in `{0, sub_m, ..., block_n - sub_m}` is non-trivial, and that set
    is the same for every S, since d depends only on BLOCK_M/BLOCK_N/CB. The two
    extra planes, `d = -sub_m` (all masked) and `d = block_n` (all kept), let the
    kernel clamp into the table instead of branching.
    """
    key = (sub_m, block_n, str(device))
    hit = _mask_bias_cache.get(key)
    if hit is not None:
        return hit
    n_pat = block_n // sub_m + 2
    i = torch.arange(sub_m, device=device).view(1, sub_m, 1)
    j = torch.arange(block_n, device=device).view(1, 1, block_n)
    d = ((torch.arange(n_pat, device=device) - 1) * sub_m).view(n_pat, 1, 1)
    bias = torch.where(i + d >= j, 0.0, -256.0).to(torch.float16).contiguous()
    _mask_bias_cache[key] = (bias, n_pat)
    return bias, n_pat


# =============================================================================
#  Autotuner
# =============================================================================
_TUNE_REPEATS = 3  # timing batches per candidate; the minimum wins


def _log_dir():
    """Where candidate compile failures are written."""
    return os.environ.get("FLAG_GEMS_FA_TLE_LOG_DIR", "./log")


def _record_failure(cfg, shape_key, exc):
    """Persist a candidate's full traceback and return the path (or None).

    A sweep skips over failures to keep going, so without this the only record
    of why a config was rejected would be the one-line summary in the log.
    """
    try:
        d = _log_dir()
        os.makedirs(d, exist_ok=True)
        path = os.path.join(
            d, f"fa_tle_autotune_fail_{cfg.label().replace('/', '_')}.log"
        )
        with open(path, "w") as f:
            f.write(
                f"config: {cfg}\nlabel: {cfg.label()}\nshape: {shape_key}\n"
                f"UB={_ub_bytes(cfg)}B L0C={_l0c_bytes(cfg)}B "
                f"workspace="
                f"{_workspace_bytes(cfg, num_cube_cores()) / 2**20:.1f}MiB\n\n"
            )
            f.write(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )
        return path
    except OSError:
        return None


def _time_config(q, k, v, cfg, is_causal, balance_causal, warmup, iters):
    """Per-launch wall time for one config, in ms. Raises on failure.

    `iters` launches are enqueued back-to-back with one `synchronize()` at each
    end, then divided. Synchronising after every launch instead would charge
    each candidate a full host round-trip, which is enough to bury the RING
    axis, whose real spread is around 0.03 ms.
    """
    import time

    sm_scale = (1.0 / q.shape[-1]) ** 0.5
    q_pre = q * sm_scale
    n = max(1, iters)

    def run():
        return flash_attention_fwd(
            q_pre,
            k,
            v,
            is_causal=is_causal,
            _q_prescaled=True,
            balance_causal=balance_causal,
            config=cfg,
        )

    for _ in range(max(1, warmup)):
        run()
    torch.npu.synchronize()

    best = float("inf")
    for _ in range(_TUNE_REPEATS):
        t0 = time.perf_counter()
        for _ in range(n):
            run()
        torch.npu.synchronize()
        best = min(best, (time.perf_counter() - t0) / n * 1e3)
    return best


def _config_is_correct(
    q, k, v, cfg, is_causal, balance_causal, ref, atol=2e-2, rtol=2e-2
):
    """True if `cfg` matches `ref`. Guards against a fast but wrong winner.

    `ref` is computed once by the caller: the fp32 reference costs more than a
    measurement at the larger shapes, so it must not be per-candidate.
    """
    out = flash_attention_fwd(
        q, k, v, is_causal=is_causal, balance_causal=balance_causal, config=cfg
    )
    return torch.allclose(ref, out, rtol=rtol, atol=atol)


class _TuneCursor:
    """Mutable cell holding the best config so far.

    Coordinate descent needs to read the running winner between candidates.
    Passing it through a cell rather than `generator.send()` keeps the drive
    loop an ordinary `for`, so the `continue` paths for pruned, wrong and failed
    candidates cannot desynchronise a handshake.
    """

    def __init__(self, base):
        self.best = base


def _tune_candidates(base, space, strategy, cursor=None):
    """Enumerate candidates, baseline first, without duplicates.

    * `coord` (default): vary one axis at a time, so the cost is
      sum(len(axis)) compiles rather than the full product. Each axis is listed
      best-known-first and the winner carries into the next axis via
      `cursor.best`, which captures the interactions that matter most (a wider
      BLOCK_N wants a different SUB_VEC_NUM, for instance). A bishengir compile
      is tens of seconds, hence the budget.
    * `grid`: the full cross product.
    """
    if strategy == "grid":
        import itertools

        keys = list(space)
        seen = set()
        for combo in itertools.product(*(space[kk] for kk in keys)):
            cfg = base._replace(**dict(zip(keys, combo)))
            if cfg not in seen:
                seen.add(cfg)
                yield cfg
        return

    seen = {base}
    yield base
    for axis, values in space.items():
        for val in values:
            cur = cursor.best if cursor is not None else base
            cfg = cur._replace(**{axis: val})
            if cfg in seen:
                continue
            seen.add(cfg)
            yield cfg


def autotune_config(
    q,
    k,
    v,
    is_causal=False,
    balance_causal=True,
    combine_batch=None,
    space=None,
    strategy="coord",
    warmup=2,
    iters=20,
    cache=True,
    cache_file=None,
    check=True,
):
    """Search for the fastest FAConfig at this shape and cache it.

    This is not `libtuner`/`triton.autotune` because BLOCK_M/BLOCK_N/CB/RING
    size the GM ring workspaces and BLOCK_M/BLOCK_N/CB decide the `task_map`
    contents: the causal schedule is built on the host from the tile shape,
    which a decorator varying constexprs against a fixed argument list cannot
    express.

    Invalid candidates are pruned by host arithmetic before a compile is spent.
    One that compiles but computes the wrong answer is rejected before it is
    timed, so it cannot win. One that fails to compile is skipped, so a bad
    corner of the space cannot abort the sweep.

    Each candidate costs a real compile plus a timed run, so this is an explicit
    offline step, never reached from the dispatch path.

    Returns the winning FAConfig.
    """
    B, Hq, S, D = q.shape
    sparse = bool(is_causal and balance_causal)
    key = (
        str(q.device),
        B,
        Hq,
        k.shape[1],
        S,
        D,
        str(q.dtype),
        bool(is_causal),
        sparse,
    )

    if cache_file:
        disk = _load_tuned(cache_file)
        if key in disk:
            _autotune_cache[key] = disk[key]
    if cache and key in _autotune_cache:
        return _autotune_cache[key]

    base = DEFAULT_CONFIG
    if combine_batch is not None:
        base = base._replace(combine_batch=combine_batch)
    space = dict(TUNE_SPACE if space is None else space)
    if combine_batch is not None:
        space.pop("combine_batch", None)  # explicitly pinned by the caller

    # A baseline that does not fit this shape (e.g. S=128 with BLOCK_N=256) is
    # replaced by the first valid point, so tuning still has a reference.
    if not config_is_valid(base, S)[0]:
        for cand in _tune_candidates(base, space, "grid"):
            if config_is_valid(cand, S)[0]:
                base = cand
                break

    # One fp32 reference for the whole search. If it cannot be built (memory,
    # unsupported shape) fall back to timing only.
    ref = None
    if check:
        try:
            ref = _reference_attention(q, k, v, is_causal)
        except Exception as e:
            logger.warning(
                "GEMS_ASCEND FA_TLE autotune: reference unavailable (%s), "
                "tuning without the correctness gate",
                type(e).__name__,
            )

    best_cfg, best_ms = None, float("inf")
    tried = skipped = failed = wrong = 0

    cursor = _TuneCursor(base)
    for cfg in _tune_candidates(base, space, strategy, cursor):
        ok, why = config_is_valid(cfg, S)
        if not ok:
            skipped += 1
            logger.debug("GEMS_ASCEND FA_TLE [skip] %s %s", cfg.label(), why)
            continue

        try:
            if ref is not None and not _config_is_correct(
                q, k, v, cfg, is_causal, balance_causal, ref
            ):
                wrong += 1
                logger.warning(
                    "GEMS_ASCEND FA_TLE [wrong] %s compiles but mismatches "
                    "the reference",
                    cfg.label(),
                )
                continue
            ms = _time_config(q, k, v, cfg, is_causal, balance_causal, warmup, iters)
        except Exception as e:  # compile or launch failure: skip, keep going
            path = _record_failure(cfg, key, e)
            failed += 1
            first = (
                str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            )
            logger.warning(
                "GEMS_ASCEND FA_TLE [fail] %s %s (full error -> %s)",
                cfg.label(),
                first[:90],
                path,
            )
            continue

        tried += 1
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
            cursor.best = cfg
        logger.debug("GEMS_ASCEND FA_TLE [ok] %s %.4f ms", cfg.label(), ms)

    if best_cfg is None:
        raise RuntimeError(
            f"fa_tle autotune: no config ran at B={B} H={Hq} S={S} D={D} "
            f"({skipped} pruned, {failed} failed to compile, {wrong} wrong; "
            f"see {_log_dir()})"
        )

    logger.info(
        "GEMS_ASCEND FA_TLE autotune: %s at %.4f ms "
        "(%d timed, %d pruned, %d failed, %d wrong)",
        best_cfg.label(),
        best_ms,
        tried,
        skipped,
        failed,
        wrong,
    )
    if cache:
        _autotune_cache[key] = best_cfg
    if cache_file:
        _store_tuned(cache_file, key, best_cfg, best_ms)
    return best_cfg


def _load_tuned(path):
    """Read a tuning cache written by `_store_tuned`. Missing or corrupt -> {}."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for entry in raw.get("entries", []):
        try:
            out[tuple(entry["key"])] = FAConfig(**entry["config"])
        except (KeyError, TypeError):
            continue
    return out


def _store_tuned(path, key, cfg, ms):
    """Merge one result into the on-disk tuning cache (best-effort)."""
    existing = _load_tuned(path)
    existing[tuple(key)] = cfg
    entries = [{"key": list(kk), "config": vv._asdict()} for kk, vv in existing.items()]
    for e in entries:
        if tuple(e["key"]) == tuple(key):
            e["ms"] = round(ms, 4)
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"version": 1, "entries": entries}, f, indent=1)
    except OSError as e:
        logger.warning(
            "GEMS_ASCEND FA_TLE could not write tuning cache %s: %s", path, e
        )


# =============================================================================
#  Kernel entry point
# =============================================================================
def flash_attention_fwd(
    q,
    k,
    v,
    combine_batch=None,
    is_causal=False,
    _q_prescaled=False,
    balance_causal=True,
    config=None,
    autotune=False,
):
    """Launch the TLE FlashAttention kernel. Returns the output tensor.

    q[B, Hq, S, D], k/v[B, Hkv, S, D], fp16, D == 64, Hq % Hkv == 0.
    `can_use_tle_fa` checks these; this function asserts them.

    `config` pins an explicit FAConfig; `autotune=True` searches for one and
    caches it per (device, shape, causal, schedule). With neither,
    DEFAULT_CONFIG is used. `combine_batch` overrides that one axis directly,
    except against a tuned config, where the tuner has already searched it.

    `balance_causal=False` runs a causal launch on the dense schedule, i.e.
    masked but without the host-side narrowing and balancing.

    `_q_prescaled=True` means the caller has already folded the softmax scale
    into q; otherwise 1/sqrt(D) is applied here.
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert D == DIM and Hq % Hkv == 0

    tuned = False
    if config is None and autotune:
        config = autotune_config(
            q,
            k,
            v,
            is_causal=is_causal,
            balance_causal=balance_causal,
            combine_batch=combine_batch,
        )
        tuned = True
    if config is None:
        config = DEFAULT_CONFIG
    if combine_batch is not None and not tuned:
        config = config._replace(combine_batch=min(combine_batch, S // config.block_n))

    block_m, block_n = config.block_m, config.block_n
    sub_vec_num, sub_m = config.sub_vec_num, config.sub_m
    ring, num_stages = config.ring, config.num_stages

    ok, why = config_is_valid(config, S)
    assert ok, (
        f"invalid config {config.label()} at S={S}: {why}. CB or SUB_VEC_NUM "
        "above their ceilings would read uninitialised rescale registers and "
        "silently corrupt the output."
    )

    num_cores = num_cube_cores()
    num_seq_blocks = S // block_m
    block_num = num_seq_blocks * Hq * B
    num_kv_blocks = S // block_n
    CB = config.combine_batch
    conbined_block_num = num_kv_blocks // CB

    block_num_per_core = block_num // num_cores
    rem_block_num = block_num % num_cores

    sparse = bool(is_causal and balance_causal)
    if sparse:
        sched_key = (q.device, CB, num_seq_blocks, Hq, B, block_m, block_n, num_cores)
        if sched_key not in _schedule_cache:
            _schedule_cache[sched_key] = _build_causal_schedule(
                num_seq_blocks,
                Hq,
                B,
                CB,
                num_kv_blocks,
                q.device,
                num_cores,
                block_m=block_m,
                block_n=block_n,
            )
        task_map, core_ntasks, max_tasks = _schedule_cache[sched_key]
    else:
        # The dense path derives everything arithmetically, but the parameters
        # must still be real tensors.
        task_map = core_ntasks = torch.zeros((1,), dtype=torch.int32, device=q.device)
        max_tasks = 1

    out = torch.empty_like(q)

    # Keyed on every ring dimension, since the tuner varies all of them.
    cache_key = (q.device, CB, ring, block_m, block_n, num_cores)
    if cache_key in _workspace_cache:
        workspace_s, workspace_p, workspace_pv = _workspace_cache[cache_key]
    else:
        workspace_s = torch.empty(
            (num_cores, ring, CB, block_m, block_n),
            dtype=torch.float16,
            device=q.device,
        )
        workspace_p = workspace_s  # Vec1 overwrites S with P in place
        workspace_pv = torch.empty(
            (num_cores, ring, CB, block_m, DIM),
            dtype=torch.float16,
            device=q.device,
        )
        _workspace_cache[cache_key] = (workspace_s, workspace_p, workspace_pv)

    sm_scale = (1.0 / D) ** 0.5
    q_scaled = q if _q_prescaled else q * sm_scale

    if is_causal:
        mask_bias, n_pat = _build_mask_bias(block_m // sub_vec_num, block_n, q.device)
    else:
        mask_bias, n_pat = _dense_mask_bias_stub(q.device), 1

    grid = (num_cores,)  # one program per AI core; one Cube + one Vector stream
    with _custom_compile_opt():
        _flash_attention_fwd_kernel[grid](
            q_scaled,
            k,
            v,
            out,
            workspace_s,
            workspace_p,
            workspace_pv,
            mask_bias,
            task_map,
            core_ntasks,
            max_tasks,
            B,
            Hq,
            Hkv,
            S,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            num_seq_blocks,
            Hq,
            Hq // Hkv,
            num_kv_blocks,
            conbined_block_num,
            block_num_per_core,
            rem_block_num,
            CB=CB,
            NUM_KV_BLOCKS=num_kv_blocks,
            IS_CAUSAL=is_causal,
            SPARSE=sparse,
            RING=ring,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            DIM=DIM,
            SUB_M=sub_m,
            SUB_VEC_NUM=sub_vec_num,
            N_PAT=n_pat,
            num_stages=num_stages,
        )
    return out


def _reference_attention(q, k, v, is_causal):
    """fp32 SDPA reference, head by head: one-shot is ~1 GB at the larger shapes.

    Only the autotuner's correctness gate uses this.
    """
    if k.shape[1] != q.shape[1]:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    ref = torch.empty_like(q)
    B, H = q.shape[0], q.shape[1]
    for b in range(B):
        for h in range(H):
            ref[b, h] = torch.nn.functional.scaled_dot_product_attention(
                q[b, h].float(),
                k[b, h].float(),
                v[b, h].float(),
                is_causal=is_causal,
            ).to(q.dtype)
    return ref


# =============================================================================
#  Dispatch guard
# =============================================================================
def can_use_tle_fa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    config=None,
):
    """True if this kernel can serve (query, key, value) exactly.

    Deliberately strict. Everything rejected here must fall back to the general
    `_attn_fwd` path in `attention.py`:

    * no attention mask and no dropout -- neither is implemented;
    * fp16 only, D == 64 -- the pipeline's tile shapes and fp16 accumulation are
      written for that one head dim;
    * square self-attention (S_q == S_k) -- the kernel carries one S;
    * S divisible by BLOCK_M and BLOCK_N -- there is no boundary masking, so a
      ragged tail would read and write out of bounds;
    * contiguous last dim -- block pointers assume unit-stride rows;
    * no gradient required -- the kernel does not emit the logsumexp that
      `scaled_dot_product_attention_backward` needs, so a tensor that will be
      differentiated has to take the general path instead.
    """
    if not HAS_TLE_DSA:
        return False
    if attn_mask is not None or dropout_p != 0.0:
        return False
    if query.dtype != torch.float16:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    if not (query.is_npu and key.is_npu and value.is_npu):
        return False

    B, Hq, Sq, D = query.shape
    Bk, Hkv, Sk, Dk = key.shape
    if key.shape != value.shape:
        return False
    if D != DIM or Dk != DIM:
        return False
    if B != Bk or Sq != Sk:
        return False
    if Hkv == 0 or Hq % Hkv != 0:
        return False
    if any(t.stride(-1) != 1 for t in (query, key, value)):
        return False

    cfg = config if config is not None else DEFAULT_CONFIG
    if not config_is_valid(cfg, Sq)[0]:
        return False

    # Backward needs the logsumexp this kernel does not produce.
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
        return False
    return True


def tle_flash_attention_forward(
    query,
    key,
    value,
    is_causal=False,
    scale=None,
    config=None,
    autotune=False,
):
    """FlashAttention forward via the TLE kernel. Call only when the guard says
    `can_use_tle_fa`.

    Returns the output tensor. `scale` defaults to 1/sqrt(D) and is folded into
    q on the host, so a caller-supplied scale costs nothing extra.
    """
    logger.debug("GEMS_ASCEND FLASH_ATTENTION_TLE_FORWARD")
    sm_scale = scale if scale is not None else 1.0 / (query.shape[-1] ** 0.5)
    q_scaled = query * sm_scale
    return flash_attention_fwd(
        q_scaled,
        key,
        value,
        is_causal=is_causal,
        _q_prescaled=True,
        config=config,
        autotune=autotune,
    )
