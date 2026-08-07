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
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import LibTuner

logger = logging.getLogger(__name__)

# All work limits below are expressed in units of tokens * padded head dim,
# the total number of rotary elements processed per launch.
# A blocked (head-parallel) config must beat the serial baseline by at least
# this relative margin before it is selected, to avoid churning tiny gains.
_AUTOTUNE_SPEEDUP_MARGIN = 0.03
# Below this work level the element count is small, so the full candidate set
# can be autotuned with negligible first-use overhead.
_FULL_AUTOTUNE_WORK_LIMIT = 8 * 1024
# Above this work level benchmark only the serial baseline plus one
# conservative blocked candidate to bound first-use tuning cost.
#
# The pruning limits only control how many candidates are measured; the final
# configuration is always picked by on-device benchmarking, so no limit
# hard-codes a device-specific choice.
_MULTI_BLOCK_AUTOTUNE_WORK_LIMIT = 64 * 1024


# Both kernels assign a complete rotary pair to one logical element, so their
# serial tiles are half as wide as the head dim and need fewer warps than a
# full-width kernel. HEAD_BLOCK_SIZE splits the head axis across programs.
WARP_CANDIDATES = {
    0: (1, 2, 4),
    1: (1, 2, 4),
    2: (2, 4),
    4: (4, 8),
    8: (4, 8),
}


def _get_rope_autotune_configs():
    return [
        triton.Config(
            {"HEAD_BLOCK_SIZE": head_block_size},
            num_warps=num_warps,
            num_stages=3,
        )
        for head_block_size, num_warps_list in WARP_CANDIDATES.items()
        for num_warps in num_warps_list
    ]


def _get_rope_inplace_autotune_configs():
    return _get_rope_autotune_configs()


def _is_baseline_config(config):
    return (
        config.kwargs["HEAD_BLOCK_SIZE"] == 0
        and config.num_warps == 4
        and config.num_stages == 3
    )


def _prune_rope_configs(configs, named_args, **_):
    baseline = next(config for config in configs if _is_baseline_config(config))
    max_heads = max(named_args["NUM_Q_HEADS"], named_args["NUM_K_HEADS"])
    n_tokens_bucket = named_args["n_tokens_bucket"]
    work = n_tokens_bucket * named_args["PADDED_HEAD_DIM"]
    if max_heads < 2:
        return [baseline]

    max_head_block_size = min(8, triton.next_power_of_2(max_heads))
    valid_configs = [
        config
        for config in configs
        if config.kwargs["HEAD_BLOCK_SIZE"] == 0
        or config.kwargs["HEAD_BLOCK_SIZE"] <= max_head_block_size
    ]
    if work >= _MULTI_BLOCK_AUTOTUNE_WORK_LIMIT:
        # At higher occupancy, benchmark only the serial baseline and one
        # conservative blocked candidate to keep first-use tuning bounded.
        preferred_head_block_size = 1 if max_heads <= 2 else min(4, max_head_block_size)
        return [
            config
            for config in valid_configs
            if _is_baseline_config(config)
            or (
                config.kwargs["HEAD_BLOCK_SIZE"] == preferred_head_block_size
                and config.num_warps == 4
            )
        ]
    if work >= _FULL_AUTOTUNE_WORK_LIMIT:
        preferred_head_block_size = 1 if max_heads <= 2 else min(4, max_head_block_size)
        candidate_head_block_sizes = {
            preferred_head_block_size,
            max_head_block_size,
        }
        return [
            config
            for config in valid_configs
            if _is_baseline_config(config)
            or config.kwargs["HEAD_BLOCK_SIZE"] in candidate_head_block_sizes
        ]
    return valid_configs


def _prune_rope_inplace_configs(configs, named_args, **_):
    max_heads = max(named_args["NUM_Q_HEADS"], named_args["NUM_K_HEADS"])
    n_tokens_bucket = named_args["n_tokens_bucket"]
    work = n_tokens_bucket * named_args["PADDED_HEAD_DIM"]
    max_head_block_size = min(8, triton.next_power_of_2(max_heads))
    valid_configs = [
        config
        for config in configs
        if config.kwargs["HEAD_BLOCK_SIZE"] == 0
        or config.kwargs["HEAD_BLOCK_SIZE"] <= max_head_block_size
    ]
    serial_configs = [
        config for config in valid_configs if config.kwargs["HEAD_BLOCK_SIZE"] == 0
    ]

    if max_heads < 2:
        return [
            config for config in valid_configs if config.kwargs["HEAD_BLOCK_SIZE"] <= 1
        ]
    if work >= _MULTI_BLOCK_AUTOTUNE_WORK_LIMIT:
        return serial_configs + [
            config
            for config in valid_configs
            if config.kwargs["HEAD_BLOCK_SIZE"] == max_head_block_size
        ]
    if work >= _FULL_AUTOTUNE_WORK_LIMIT:
        preferred_head_block_size = 1 if max_heads <= 2 else min(4, max_head_block_size)
        candidate_head_block_sizes = {
            preferred_head_block_size,
            max_head_block_size,
        }
        return serial_configs + [
            config
            for config in valid_configs
            if config.kwargs["HEAD_BLOCK_SIZE"] in candidate_head_block_sizes
        ]
    return valid_configs


class _RopeTuner(LibTuner):
    @staticmethod
    def select_config(configs, timings):
        best_config = min(configs, key=lambda config: timings[config][0])
        baseline = next(
            (config for config in configs if _is_baseline_config(config)),
            None,
        )
        if baseline is None or best_config is baseline:
            return best_config, timings

        best_time = timings[best_config][0]
        baseline_time = timings[baseline][0]
        if best_time <= baseline_time * (1.0 - _AUTOTUNE_SPEEDUP_MARGIN):
            return best_config, timings
        return baseline, timings

    @staticmethod
    def policy(bench_fn, configs, args, kwargs):
        del args, kwargs
        configs = list(configs)
        if len(configs) == 1:
            return configs[0], {}

        timings = {config: bench_fn(config) for config in configs}
        return _RopeTuner.select_config(configs, timings)


class _RopeInplaceTuner(_RopeTuner):
    def policy(self, bench_fn, configs, args, kwargs):
        configs = list(configs)
        if len(configs) == 1:
            return configs[0], {}

        # A standard restore_value hook clones and copies every mutated tensor
        # on every timed launch. That overhead hides kernel differences for
        # large q/k tensors. Back up once, benchmark without per-launch
        # restoration, then restore before running the selected config.
        backups = (args[0].detach().clone(), args[1].detach().clone())
        try:
            timings = {config: bench_fn(config) for config in configs}
        finally:
            with torch.no_grad():
                args[0].copy_(backups[0])
                args[1].copy_(backups[1])
        return self.select_config(configs, timings)


_COMMON_AUTOTUNE_KEYS = [
    "n_tokens_bucket",
    "NUM_Q_HEADS",
    "NUM_K_HEADS",
    "HEAD_DIM",
    "PADDED_HEAD_DIM",
    "ROTARY_INTERLEAVED",
    "q_stride_s",
    "q_stride_h",
    "q_stride_d",
    "k_stride_s",
    "k_stride_h",
    "k_stride_d",
    "p_stride_s",
    "cos_stride_s",
    "sin_stride_s",
]


def _rope_grid(n_tokens):
    def grid(meta):
        num_head_blocks = (
            triton.cdiv(
                max(meta["NUM_Q_HEADS"], meta["NUM_K_HEADS"]),
                meta["HEAD_BLOCK_SIZE"],
            )
            if meta["HEAD_BLOCK_SIZE"] > 0
            else 1
        )
        return n_tokens, num_head_blocks

    return grid


@libentry()
@libtuner(
    configs=_get_rope_autotune_configs(),
    key=_COMMON_AUTOTUNE_KEYS
    + [
        "oq_stride_s",
        "oq_stride_h",
        "oq_stride_d",
        "ok_stride_s",
        "ok_stride_h",
        "ok_stride_d",
    ],
    warmup=5,
    rep=10,
    prune_configs_by={"early_config_prune": _prune_rope_configs},
    policy=_RopeTuner,
)
@triton.jit
def apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    n_tokens_bucket,  # tuning/cache key only; the grid uses the exact token count
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    HEAD_BLOCK_SIZE: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = ext.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    # Assign both values of a rotary pair to the same logical lane, mirroring
    # the in-place kernel. Each input value is loaded exactly once (instead of
    # once here and again as the partner's rotated element), halves cos/sin
    # traffic, and keeps the same code shape shared by both entry points.
    pair_block = tl.arange(0, PADDED_HEAD_DIM // 2)
    pair_mask = pair_block < HEAD_DIM // 2
    if ROTARY_INTERLEAVED:
        first_dim = pair_block * 2
        second_dim = first_dim + 1
    else:
        first_dim = pair_block
        second_dim = pair_block + HEAD_DIM // 2

    cos = tl.load(cos_ptr + pair_block, mask=pair_mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + pair_block, mask=pair_mask, other=0.0).to(tl.float32)

    oq_ptr += s_id * oq_stride_s
    q_ptr += s_id * q_stride_s
    ok_ptr += s_id * ok_stride_s
    k_ptr += s_id * k_stride_s

    if HEAD_BLOCK_SIZE > 0:
        # Split the head axis across programs to expose enough parallelism for
        # decode and other small-token workloads. HEAD_BLOCK_SIZE balances
        # inter-program parallelism against cos/sin reuse.
        head_block_start = ext.program_id(1) * HEAD_BLOCK_SIZE
        head_offsets = head_block_start + tl.arange(0, HEAD_BLOCK_SIZE)

        if head_block_start < NUM_Q_HEADS:
            q_mask = (head_offsets[:, None] < NUM_Q_HEADS) & pair_mask[None, :]
            q_first_cols = (
                head_offsets[:, None] * q_stride_h + first_dim[None, :] * q_stride_d
            )
            q_second_cols = (
                head_offsets[:, None] * q_stride_h + second_dim[None, :] * q_stride_d
            )
            oq_first_cols = (
                head_offsets[:, None] * oq_stride_h + first_dim[None, :] * oq_stride_d
            )
            oq_second_cols = (
                head_offsets[:, None] * oq_stride_h + second_dim[None, :] * oq_stride_d
            )
            x0 = tl.load(q_ptr + q_first_cols, mask=q_mask, other=0.0)
            x1 = tl.load(q_ptr + q_second_cols, mask=q_mask, other=0.0)
            tl.store(
                oq_ptr + oq_first_cols,
                x0 * cos[None, :] - x1 * sin[None, :],
                mask=q_mask,
            )
            tl.store(
                oq_ptr + oq_second_cols,
                x1 * cos[None, :] + x0 * sin[None, :],
                mask=q_mask,
            )

        if head_block_start < NUM_K_HEADS:
            k_mask = (head_offsets[:, None] < NUM_K_HEADS) & pair_mask[None, :]
            k_first_cols = (
                head_offsets[:, None] * k_stride_h + first_dim[None, :] * k_stride_d
            )
            k_second_cols = (
                head_offsets[:, None] * k_stride_h + second_dim[None, :] * k_stride_d
            )
            ok_first_cols = (
                head_offsets[:, None] * ok_stride_h + first_dim[None, :] * ok_stride_d
            )
            ok_second_cols = (
                head_offsets[:, None] * ok_stride_h + second_dim[None, :] * ok_stride_d
            )
            k_first = tl.load(k_ptr + k_first_cols, mask=k_mask, other=0.0)
            k_second = tl.load(k_ptr + k_second_cols, mask=k_mask, other=0.0)
            tl.store(
                ok_ptr + ok_first_cols,
                k_first * cos[None, :] - k_second * sin[None, :],
                mask=k_mask,
            )
            tl.store(
                ok_ptr + ok_second_cols,
                k_second * cos[None, :] + k_first * sin[None, :],
                mask=k_mask,
            )
    else:
        for off_h in range(0, NUM_Q_HEADS):
            first_cols = off_h * q_stride_h + (first_dim * q_stride_d)
            second_cols = off_h * q_stride_h + (second_dim * q_stride_d)
            o_first_cols = off_h * oq_stride_h + (first_dim * oq_stride_d)
            o_second_cols = off_h * oq_stride_h + (second_dim * oq_stride_d)

            x0 = tl.load(q_ptr + first_cols, mask=pair_mask, other=0.0)
            x1 = tl.load(q_ptr + second_cols, mask=pair_mask, other=0.0)
            tl.store(
                oq_ptr + o_first_cols,
                x0 * cos - x1 * sin,
                mask=pair_mask,
            )
            tl.store(
                oq_ptr + o_second_cols,
                x1 * cos + x0 * sin,
                mask=pair_mask,
            )

        for off_h in range(0, NUM_K_HEADS):
            first_cols = off_h * k_stride_h + (first_dim * k_stride_d)
            second_cols = off_h * k_stride_h + (second_dim * k_stride_d)
            o_first_cols = off_h * ok_stride_h + (first_dim * ok_stride_d)
            o_second_cols = off_h * ok_stride_h + (second_dim * ok_stride_d)

            x0 = tl.load(k_ptr + first_cols, mask=pair_mask, other=0.0)
            x1 = tl.load(k_ptr + second_cols, mask=pair_mask, other=0.0)
            tl.store(
                ok_ptr + o_first_cols,
                x0 * cos - x1 * sin,
                mask=pair_mask,
            )
            tl.store(
                ok_ptr + o_second_cols,
                x1 * cos + x0 * sin,
                mask=pair_mask,
            )


@libentry()
@libtuner(
    configs=_get_rope_inplace_autotune_configs(),
    key=_COMMON_AUTOTUNE_KEYS,
    warmup=5,
    rep=10,
    prune_configs_by={"early_config_prune": _prune_rope_inplace_configs},
    policy=_RopeInplaceTuner,
)
@triton.jit
def apply_rotary_pos_emb_inplace_kernel(
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    n_tokens_bucket,  # tuning/cache key only; the grid uses the exact token count
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    HEAD_BLOCK_SIZE: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = ext.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    # Assign both values of a rotary pair to the same logical lane. Both input
    # values are therefore loaded into registers before either output is
    # stored, avoiding cross-lane load/store hazards in the in-place path.
    pair_block = tl.arange(0, PADDED_HEAD_DIM // 2)
    pair_mask = pair_block < HEAD_DIM // 2
    if ROTARY_INTERLEAVED:
        first_dim = pair_block * 2
        second_dim = first_dim + 1
    else:
        first_dim = pair_block
        second_dim = pair_block + HEAD_DIM // 2

    cos = tl.load(cos_ptr + pair_block, mask=pair_mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + pair_block, mask=pair_mask, other=0.0).to(tl.float32)

    q_ptr += s_id * q_stride_s
    k_ptr += s_id * k_stride_s

    if HEAD_BLOCK_SIZE > 0:
        head_block_start = ext.program_id(1) * HEAD_BLOCK_SIZE
        head_offsets = head_block_start + tl.arange(0, HEAD_BLOCK_SIZE)

        if head_block_start < NUM_Q_HEADS:
            q_mask = (head_offsets[:, None] < NUM_Q_HEADS) & pair_mask[None, :]
            q_first_cols = (
                head_offsets[:, None] * q_stride_h + first_dim[None, :] * q_stride_d
            )
            q_second_cols = (
                head_offsets[:, None] * q_stride_h + second_dim[None, :] * q_stride_d
            )
            q_first = tl.load(q_ptr + q_first_cols, mask=q_mask, other=0.0)
            q_second = tl.load(q_ptr + q_second_cols, mask=q_mask, other=0.0)
            tl.store(
                q_ptr + q_first_cols,
                q_first * cos[None, :] - q_second * sin[None, :],
                mask=q_mask,
            )
            tl.store(
                q_ptr + q_second_cols,
                q_second * cos[None, :] + q_first * sin[None, :],
                mask=q_mask,
            )

        if head_block_start < NUM_K_HEADS:
            k_mask = (head_offsets[:, None] < NUM_K_HEADS) & pair_mask[None, :]
            k_first_cols = (
                head_offsets[:, None] * k_stride_h + first_dim[None, :] * k_stride_d
            )
            k_second_cols = (
                head_offsets[:, None] * k_stride_h + second_dim[None, :] * k_stride_d
            )
            k_first = tl.load(k_ptr + k_first_cols, mask=k_mask, other=0.0)
            k_second = tl.load(k_ptr + k_second_cols, mask=k_mask, other=0.0)
            tl.store(
                k_ptr + k_first_cols,
                k_first * cos[None, :] - k_second * sin[None, :],
                mask=k_mask,
            )
            tl.store(
                k_ptr + k_second_cols,
                k_second * cos[None, :] + k_first * sin[None, :],
                mask=k_mask,
            )
    else:
        for off_h in range(0, NUM_Q_HEADS):
            first_cols = off_h * q_stride_h + first_dim * q_stride_d
            second_cols = off_h * q_stride_h + second_dim * q_stride_d

            q_first = tl.load(q_ptr + first_cols, mask=pair_mask, other=0.0)
            q_second = tl.load(q_ptr + second_cols, mask=pair_mask, other=0.0)
            tl.store(
                q_ptr + first_cols,
                q_first * cos - q_second * sin,
                mask=pair_mask,
            )
            tl.store(
                q_ptr + second_cols,
                q_second * cos + q_first * sin,
                mask=pair_mask,
            )

        for off_h in range(0, NUM_K_HEADS):
            first_cols = off_h * k_stride_h + first_dim * k_stride_d
            second_cols = off_h * k_stride_h + second_dim * k_stride_d

            k_first = tl.load(k_ptr + first_cols, mask=pair_mask, other=0.0)
            k_second = tl.load(k_ptr + second_cols, mask=pair_mask, other=0.0)
            tl.store(
                k_ptr + first_cols,
                k_first * cos - k_second * sin,
                mask=pair_mask,
            )
            tl.store(
                k_ptr + second_cols,
                k_second * cos + k_first * sin,
                mask=pair_mask,
            )


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
):
    """
    Apply rotary position embedding to q and k

    Args:
        q: (*, q_heads, head_dim)
        k: (*, k_heads, head_dim)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
        position_ids: (*, ), optional, position ids for each token
        rotary_interleaved: whether the head_dim is rotated in an interleaved way

    Returns:
        q_embed: (*, q_heads, head_dim)
        k_embed: (*, k_heads, head_dim)
    """
    logger.debug("GEMS ROTARY_POS_EMBEDDING")
    assert (
        k.shape[-1] == q.shape[-1]
    ), f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
    assert (
        cos.shape[-1] == sin.shape[-1]
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == q.shape[-1]
    ), f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    q_shape = q.shape
    k_shape = k.shape

    assert (
        q.shape[:-2] == k.shape[:-2]
    ), f"q and k must have the same length, got {q.shape[:-2]} and {k.shape[:-2]}"
    if position_ids is None:
        assert (
            len(q.shape) == 4
        ), f"q must have 4 dimensions if position_ids is not provided, got {q.shape}"
        seq_len = q.shape[-3]
    else:
        assert (
            position_ids.shape == q.shape[:-2]
        ), f"position_ids must have the same length as q, got {position_ids.shape} and {q.shape[:-2]}"

        position_ids = position_ids.view(-1)
        seq_len = None

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    n_tokens, q_heads, head_dim = q.shape
    k_heads = k.shape[-2]
    # Keep the bucket as a runtime argument: LibEntry can cache the selected
    # config per workload class without specializing kernel code on exact N.
    n_tokens_bucket = max(triton.next_power_of_2(n_tokens), 1)

    # The block size must be the next power of two, sometimes we need to pad it.
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    if inplace:
        kernel_args = (
            q,
            k,
            cos,
            sin,
            position_ids,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            position_ids.stride(0) if position_ids is not None else 0,
            cos.stride(0),
            sin.stride(0),
            seq_len,
            n_tokens_bucket,
            q_heads,
            k_heads,
            head_dim,
            padded_head_dim,
        )
        with torch_device_fn.device(q.device):
            apply_rotary_pos_emb_inplace_kernel[_rope_grid(n_tokens)](
                *kernel_args,
                ROTARY_INTERLEAVED=rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
            )
        return q.view(q_shape), k.view(k_shape)
    # If not inplace, we need to create new tensors for q_embed and k_embed
    else:
        q_embed = torch.empty_like(q)
        k_embed = torch.empty_like(k)

        kernel_args = (
            q_embed,
            k_embed,
            q,
            k,
            cos,
            sin,
            position_ids,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            q_embed.stride(0),
            q_embed.stride(1),
            q_embed.stride(2),
            k_embed.stride(0),
            k_embed.stride(1),
            k_embed.stride(2),
            position_ids.stride(0) if position_ids is not None else 0,
            cos.stride(0),
            sin.stride(0),
            seq_len,
            n_tokens_bucket,
            q_heads,
            k_heads,
            head_dim,
            padded_head_dim,
        )
        with torch_device_fn.device(q_embed.device):
            apply_rotary_pos_emb_kernel[_rope_grid(n_tokens)](
                *kernel_args,
                ROTARY_INTERLEAVED=rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
            )
        q_embed = q_embed.view(q_shape)
        k_embed = k_embed.view(k_shape)
        return q_embed, k_embed
