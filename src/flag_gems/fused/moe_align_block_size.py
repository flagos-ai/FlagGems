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
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.utils import has_triton_tle, libentry, libtuner

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle
        import triton.experimental.tle.language.gpu as tleg

        HAS_TLE = True
    except ImportError:
        tle = None
        tleg = None
        HAS_TLE = False
else:
    tle = None
    tleg = None
    HAS_TLE = False


logger = logging.getLogger(__name__)

TLE_CLUSTER_SIZE = 8
TLE_BIG_TOKEN_THRESHOLD_TOKENS = 4096
COMPACT_ALIGN_MAX_ROUTES = 1024
COMPACT_ALIGN_MIN_EXPERTS = 256
COMPACT_ALIGN_MAX_EXPERTS = 512
COMPACT_ALIGN_BLOCK_SIZE = 16
COMPACT_ALIGN_TLE_OVERRIDE_ROUTES = 640
COMPACT_ALIGN_TLE_OVERRIDE_EXPERTS = 512
EP_COMPACT_GLOBAL_EXPERTS = 288
EP_COMPACT_LOCAL_EXPERTS = 18
EP_COMPACT_MAX_DECODE_ROUTES = 1024
EP_ROUTE_BLOCK_SIZE = 16
EP_ROUTE_BLOCK_MAX_ROUTES = 64
_TRITON_ALLOCATOR_INSTALLED = False
TLE_ATOMIC_WARPS_CONFIGS = [
    triton.Config(kwargs={}, num_warps=4),
    triton.Config(kwargs={}, num_warps=8),
]
TLE_CLUSTER_LAUNCH_CONFIGS = [
    triton.Config(kwargs={"BLOCK_TOKENS": 128}, num_warps=4),
    triton.Config(kwargs={"BLOCK_TOKENS": 128}, num_warps=8),
    triton.Config(kwargs={"BLOCK_TOKENS": 256}, num_warps=4),
    triton.Config(kwargs={"BLOCK_TOKENS": 256}, num_warps=8),
    triton.Config(kwargs={"BLOCK_TOKENS": 512}, num_warps=4),
    triton.Config(kwargs={"BLOCK_TOKENS": 512}, num_warps=8),
    triton.Config(kwargs={"BLOCK_TOKENS": 1024}, num_warps=4),
    triton.Config(kwargs={"BLOCK_TOKENS": 1024}, num_warps=8),
]


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@lru_cache(maxsize=64)
def _block_mesh(num_blocks: int):
    return tle.device_mesh({"block": [("block_x", int(num_blocks))]})


@lru_cache(maxsize=1)
def _block_cluster_mesh_8():
    return tle.device_mesh({"block_cluster": [("cluster_x", TLE_CLUSTER_SIZE)]})


def _supports_tle_cluster_remote() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 9


@lru_cache(maxsize=None)
def _supports_compact_align(device_index: int | None = None) -> bool:
    """Restrict the measured compact path to NVIDIA SM90 devices."""
    if runtime_device.vendor_name != "nvidia" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device_index) == (9, 0)


def _install_triton_default_allocator(device: torch.device) -> None:
    global _TRITON_ALLOCATOR_INSTALLED
    if _TRITON_ALLOCATOR_INSTALLED:
        return

    def _alloc(size: int, _alignment: int, _stream: Optional[int]):
        return torch.empty((size,), dtype=torch.uint8, device=device)

    triton.set_allocator(_alloc)
    _TRITON_ALLOCATOR_INSTALLED = True


def _pick_tle_fused_launch_params(numel: int, num_experts: int) -> "tuple[int, int]":
    if num_experts >= 256:
        if numel >= 32768:
            return 4096, 4
        if numel >= 1024:
            return 1024, 4
        return 256, 8

    if numel <= 512:
        return 128, 8
    if num_experts <= 64 and numel <= 2048:
        return 128, 8
    return 256, 8


def _pick_tle_atomic_fused_launch_params(
    numel: int, num_experts: int
) -> "tuple[int, int]":
    if num_experts >= 256:
        if numel <= 16384:
            return 256, 8
        if numel <= 32768:
            return 512, 4
        return 1024, 4
    return _pick_tle_fused_launch_params(numel, num_experts)


def _pick_tle_atomic_fused_num_blocks(
    numel: int, num_experts: int, block_tokens: int, device: torch.device
) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 1
    props = torch.cuda.get_device_properties(device)
    sm_count = int(getattr(props, "multi_processor_count", 1))
    token_programs = triton.cdiv(numel, block_tokens)
    cap_mult = 4 if num_experts < 256 else 16
    block_cap = sm_count * cap_mult
    return max(1, min(token_programs, block_cap))


@libentry()
@libtuner(
    configs=TLE_ATOMIC_WARPS_CONFIGS,
    key=["numel"],
    strategy=["log"],
)
@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_tle_atomic_fused_coop(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    cumsum_ptr,
    mesh: tl.constexpr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    EXPERTS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < num_experts
    token_offsets = tl.arange(0, BLOCK_TOKENS)

    for base in range(
        pid * BLOCK_TOKENS, numel_sorted_token_ids, NUM_BLOCKS * BLOCK_TOKENS
    ):
        offs = base + token_offsets
        tl.store(sorted_token_ids_ptr + offs, numel, mask=offs < numel_sorted_token_ids)
    for base in range(pid * BLOCK_TOKENS, numel_expert_ids, NUM_BLOCKS * BLOCK_TOKENS):
        offs = base + token_offsets
        tl.store(expert_ids_ptr + offs, -1, mask=offs < numel_expert_ids)
    if pid == 0:
        tl.store(cumsum_ptr + expert_offsets, 0, mask=expert_mask)
    tle.distributed_barrier(mesh)

    local_counts = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_counts_ptrs = tle.gpu.local_ptr(local_counts, (expert_offsets,))
    tl.store(local_counts_ptrs, 0, mask=expert_mask)

    for base in range(pid * BLOCK_TOKENS, numel, NUM_BLOCKS * BLOCK_TOKENS):
        offs = base + token_offsets
        mask = offs < numel
        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        count_ptrs = tle.gpu.local_ptr(local_counts, (expert_id,))
        tl.atomic_add(count_ptrs, 1, mask=mask, sem="relaxed", scope="cta")

    local_counts_vals = tl.load(local_counts_ptrs, mask=expert_mask, other=0)
    prefix_before = tl.atomic_add(
        cumsum_ptr + expert_offsets,
        local_counts_vals,
        mask=expert_mask,
        sem="acq_rel",
        scope="gpu",
    )
    tl.store(local_counts_ptrs, prefix_before, mask=expert_mask)
    tle.distributed_barrier(mesh)

    if pid == 0:
        total_counts = tl.load(cumsum_ptr + expert_offsets, mask=expert_mask, other=0)
        aligned_counts = tl.cdiv(total_counts, block_size) * block_size
        expert_starts = tl.cumsum(aligned_counts, axis=0) - aligned_counts
        tl.store(cumsum_ptr + expert_offsets, expert_starts, mask=expert_mask)
        total_tokens = tl.sum(aligned_counts, axis=0)
        tl.store(num_tokens_post_pad_ptr, total_tokens)
    tle.distributed_barrier(mesh)

    expert_starts_local = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    expert_starts_ptrs = tle.gpu.local_ptr(expert_starts_local, (expert_offsets,))
    expert_starts_vals = tl.load(cumsum_ptr + expert_offsets, mask=expert_mask, other=0)
    tl.store(expert_starts_ptrs, expert_starts_vals, mask=expert_mask)

    total_tokens = tl.load(num_tokens_post_pad_ptr)
    for local_expert_idx in range(EXPERTS_PER_PROG):
        expert_id = pid + local_expert_idx * NUM_BLOCKS
        valid_expert = expert_id < num_experts
        start_idx = tl.load(
            tle.gpu.local_ptr(expert_starts_local, (expert_id,)),
            mask=valid_expert,
            other=0,
        )
        next_expert = expert_id + 1
        has_next = valid_expert & (next_expert < num_experts)
        end_idx = tl.load(
            tle.gpu.local_ptr(expert_starts_local, (next_expert,)),
            mask=has_next,
            other=total_tokens,
        )
        end_idx = tl.where(has_next, end_idx, total_tokens)
        start_idx = tl.where(valid_expert, start_idx, 0)
        end_idx = tl.where(valid_expert, end_idx, 0)
        for i in range(start_idx, end_idx, block_size):
            tl.store(expert_ids_ptr + i // block_size, expert_id)

    for base in range(pid * BLOCK_TOKENS, numel, NUM_BLOCKS * BLOCK_TOKENS):
        offs = base + token_offsets
        mask = offs < numel
        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        count_ptrs = tle.gpu.local_ptr(local_counts, (expert_id,))
        rank_with_prefix = tl.atomic_add(
            count_ptrs, 1, mask=mask, sem="relaxed", scope="cta"
        )
        rank_base = tl.load(
            tle.gpu.local_ptr(expert_starts_local, (expert_id,)), mask=mask, other=0
        )
        rank_post_pad = rank_with_prefix + rank_base
        tl.store(sorted_token_ids_ptr + rank_post_pad, offs, mask=mask)


@libentry()
@libtuner(
    configs=TLE_CLUSTER_LAUNCH_CONFIGS,
    key=["numel"],
    strategy=["log"],
)
@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_tle_cluster_fused(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    mesh: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    EXPERTS_PER_SHARD: tl.constexpr,
):
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    is_rank0 = cluster_rank == 0
    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < num_experts

    init_offsets = tl.arange(0, BLOCK_TOKENS)
    for base in range(
        cluster_rank * BLOCK_TOKENS, numel_sorted_token_ids, CLUSTER_SIZE * BLOCK_TOKENS
    ):
        offs = base + init_offsets
        mask = offs < numel_sorted_token_ids
        tl.store(sorted_token_ids_ptr + offs, numel, mask=mask)
    for base in range(
        cluster_rank * BLOCK_TOKENS, numel_expert_ids, CLUSTER_SIZE * BLOCK_TOKENS
    ):
        offs = base + init_offsets
        mask = offs < numel_expert_ids
        tl.store(expert_ids_ptr + offs, -1, mask=mask)

    local_counts = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    cumsum_local = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    rank0_cumsum_ptrs = tle.gpu.local_ptr(cumsum_local, (expert_offsets,))
    if is_rank0:
        tl.store(rank0_cumsum_ptrs, 0, mask=expert_mask)
    tle.distributed_barrier(mesh)

    local_counts_ptrs = tle.gpu.local_ptr(local_counts, (expert_offsets,))
    tl.store(local_counts_ptrs, 0, mask=expert_mask)

    for base in range(cluster_rank * BLOCK_TOKENS, numel, CLUSTER_SIZE * BLOCK_TOKENS):
        offs = base + init_offsets
        mask = offs < numel
        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        count_ptrs = tle.gpu.local_ptr(local_counts, (expert_id,))
        tl.atomic_add(count_ptrs, 1, mask=mask, sem="relaxed", scope="cta")

    local_counts_vals = tl.load(local_counts_ptrs, mask=expert_mask, other=0)
    rank0_cumsum_remote = tle.remote(cumsum_local, 0, scope=mesh)
    rank0_cumsum_remote_ptrs = tle.gpu.local_ptr(rank0_cumsum_remote, (expert_offsets,))
    prefix_before = tl.atomic_add(
        rank0_cumsum_remote_ptrs,
        local_counts_vals,
        mask=expert_mask,
        sem="relaxed",
        scope="cta",
    )
    tl.store(local_counts_ptrs, prefix_before, mask=expert_mask)

    tle.distributed_barrier(mesh)

    if is_rank0:
        total_counts = tl.load(rank0_cumsum_ptrs, mask=expert_mask, other=0)
        aligned_counts = tl.cdiv(total_counts, block_size) * block_size
        expert_cumsum_inclusive = tl.cumsum(aligned_counts, axis=0)
        expert_start_offsets = expert_cumsum_inclusive - aligned_counts
        tl.store(rank0_cumsum_ptrs, expert_start_offsets, mask=expert_mask)
        total_tokens = tl.sum(aligned_counts, axis=0)
        tl.store(num_tokens_post_pad_ptr, total_tokens)

    tle.distributed_barrier(mesh)

    rank0_cumsum_remote = tle.remote(cumsum_local, 0, scope=mesh)
    rank0_cumsum_remote_ptrs = tle.gpu.local_ptr(rank0_cumsum_remote, (expert_offsets,))
    cumsum_vals = tl.load(rank0_cumsum_remote_ptrs, mask=expert_mask, other=0)
    tl.store(
        tle.gpu.local_ptr(cumsum_local, (expert_offsets,)),
        cumsum_vals,
        mask=expert_mask,
    )
    total_tokens = tl.load(num_tokens_post_pad_ptr)

    for local_expert_idx in range(EXPERTS_PER_SHARD):
        expert_idx = cluster_rank * EXPERTS_PER_SHARD + local_expert_idx
        expert_id = expert_idx
        valid_expert = expert_id < num_experts
        start_ptr = tle.gpu.local_ptr(cumsum_local, (expert_id,))
        start_idx = tl.load(start_ptr, mask=valid_expert, other=0)
        next_expert_id = expert_id + 1
        has_next = valid_expert & (next_expert_id < num_experts)
        next_ptr = tle.gpu.local_ptr(cumsum_local, (next_expert_id,))
        end_from_next = tl.load(next_ptr, mask=has_next, other=0)
        end_idx = tl.where(has_next, end_from_next, total_tokens)
        start_idx = tl.where(valid_expert, start_idx, 0)
        end_idx = tl.where(valid_expert, end_idx, 0)
        for i in range(start_idx, end_idx, block_size):
            tl.store(expert_ids_ptr + i // block_size, expert_idx)

    tle.distributed_barrier(mesh)

    for base in range(cluster_rank * BLOCK_TOKENS, numel, CLUSTER_SIZE * BLOCK_TOKENS):
        offs = base + init_offsets
        mask = offs < numel
        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        count_ptrs = tle.gpu.local_ptr(local_counts, (expert_id,))
        rank_with_prefix = tl.atomic_add(
            count_ptrs, 1, mask=mask, sem="relaxed", scope="cta"
        )
        base_ptrs = tle.gpu.local_ptr(cumsum_local, (expert_id,))
        rank_base = tl.load(base_ptrs, mask=mask, other=0)
        rank_post_pad = rank_with_prefix + rank_base
        tl.store(sorted_token_ids_ptr + rank_post_pad, offs, mask=mask)


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_expert: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, -1, mask=mask_expert)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    offsets = start_idx + tl.arange(0, tokens_per_thread)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(tokens_cnts_ptr + off_c + expert_id, 1, mask=mask)


@triton.jit
def moe_align_block_size_stage2_vec(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    offset = tl.arange(0, num_experts) + 1
    token_cnt = tl.load(tokens_cnts_ptr + offset * num_experts + pid)
    cnt = tl.cumsum(token_cnt, axis=0)
    tl.store(tokens_cnts_ptr + offset * num_experts + pid, cnt)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    num_experts_next_power_of_2: tl.constexpr,
    block_size: tl.constexpr,
):
    off_cnt = num_experts * num_experts

    expert_offsets = tl.arange(0, num_experts_next_power_of_2)
    mask = expert_offsets < num_experts
    token_cnts = tl.load(tokens_cnts_ptr + off_cnt + expert_offsets, mask=mask)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size

    cumsum_values = tl.cumsum(aligned_cnts, axis=0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum_values, mask=mask)

    total_tokens = tl.sum(aligned_cnts, axis=0)
    tl.store(total_tokens_post_pad_ptr, total_tokens)


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    offset = tl.arange(0, tokens_per_thread) + start_idx
    mask = offset < numel
    expert_id = tl.load(topk_ids_ptr + offset, mask=mask)
    token_idx_in_expert = tl.atomic_add(
        tokens_cnts_ptr + off_t + expert_id, 1, mask=mask
    )
    rank_post_pad = token_idx_in_expert + tl.load(cumsum_ptr + expert_id, mask=mask)
    tl.store(sorted_token_ids_ptr + rank_post_pad, offset, mask=mask)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    logger.debug("GEMS MOE ALIGN BLOCK SIZE")
    numel = topk_ids.numel()
    numel_sorted_token_ids = sorted_token_ids.numel()
    numel_expert_ids = expert_ids.numel()
    grid = (num_experts,)
    tokens_per_thread = triton.next_power_of_2(ceil_div(numel, num_experts))
    block_size_sorted = triton.next_power_of_2(
        ceil_div(numel_sorted_token_ids, num_experts)
    )
    block_size_expert = triton.next_power_of_2(ceil_div(numel_expert_ids, num_experts))
    block_expert_tle = triton.next_power_of_2(num_experts)

    if HAS_TLE and topk_ids.is_cuda and block_expert_tle <= 1024:
        block_tokens_taf, _ = _pick_tle_atomic_fused_launch_params(numel, num_experts)
        experts_per_shard = ceil_div(num_experts, TLE_CLUSTER_SIZE)
        num_tokens = topk_ids.shape[0] if topk_ids.ndim > 1 else numel

        def _run_tle_atomic_fused() -> bool:
            cumsum_tle = torch.zeros(
                (num_experts,), dtype=torch.int32, device=topk_ids.device
            )
            num_blocks = _pick_tle_atomic_fused_num_blocks(
                numel,
                num_experts,
                block_tokens_taf,
                topk_ids.device,
            )
            experts_per_prog = ceil_div(num_experts, num_blocks)
            while True:
                try:
                    moe_align_block_size_tle_atomic_fused_coop[(num_blocks,)](
                        topk_ids,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_pad,
                        cumsum_tle,
                        _block_mesh(num_blocks),
                        num_experts,
                        block_size,
                        numel,
                        numel_sorted_token_ids,
                        numel_expert_ids,
                        NUM_BLOCKS=num_blocks,
                        BLOCK_TOKENS=block_tokens_taf,
                        BLOCK_EXPERT=block_expert_tle,
                        EXPERTS_PER_PROG=experts_per_prog,
                        launch_cooperative_grid=True,
                    )
                    return True
                except Exception as ex:
                    msg = str(ex).lower()
                    if "no allocator was set" in msg:
                        _install_triton_default_allocator(topk_ids.device)
                        continue
                    if num_blocks <= 1 or "cooperative" not in msg:
                        logger.debug(
                            "TLE atomic fused launch failed, fallback to triton: %s",
                            ex,
                        )
                        return False
                    num_blocks = max(1, num_blocks // 2)
                    experts_per_prog = ceil_div(num_experts, num_blocks)

        if (
            num_tokens < TLE_BIG_TOKEN_THRESHOLD_TOKENS
            and _supports_tle_cluster_remote()
        ):
            try:
                moe_align_block_size_tle_cluster_fused[(1,)](
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_pad,
                    num_experts,
                    block_size,
                    numel,
                    numel_sorted_token_ids,
                    numel_expert_ids,
                    mesh=_block_cluster_mesh_8(),
                    CLUSTER_SIZE=TLE_CLUSTER_SIZE,
                    BLOCK_EXPERT=block_expert_tle,
                    EXPERTS_PER_SHARD=experts_per_shard,
                )
                return
            except Exception as ex:
                logger.debug(
                    "TLE cluster fused launch failed, fallback to atomic/triton: %s",
                    ex,
                )

        if _run_tle_atomic_fused():
            return

    # The tensor needs to be padded before calculating IDs,
    # to prevent out-of-bounds address access.
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    num_experts_next_power_of_2 = triton.next_power_of_2(num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
        sorted_token_ids,
        expert_ids,
        numel_sorted_token_ids,
        numel_expert_ids,
        block_size_sorted,
        block_size_expert,
    )
    if num_experts == triton.next_power_of_2(num_experts):
        moe_align_block_size_stage2_vec[grid](tokens_cnts, num_experts)
    else:
        moe_align_block_size_stage2[grid](tokens_cnts, num_experts)
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        num_experts_next_power_of_2,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )


@triton.jit
def _moe_align_block_size_singleton_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    num_routes: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    route_idx = tl.program_id(0)
    offsets = route_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    values = tl.full((BLOCK_SIZE_M,), num_routes, dtype=tl.int32)
    values = tl.where(tl.arange(0, BLOCK_SIZE_M) == 0, route_idx, values)
    tl.store(sorted_token_ids_ptr + offsets, values)
    tl.store(expert_ids_ptr + route_idx, tl.load(topk_ids_ptr + route_idx))
    if route_idx == 0:
        tl.store(num_tokens_post_pad_ptr, num_routes * BLOCK_SIZE_M)


def moe_align_block_size_singleton(
    topk_ids: torch.Tensor,
    block_size: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    num_routes = topk_ids.numel()
    sorted_token_ids = torch.empty(
        (num_routes * block_size,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty((num_routes,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    _moe_align_block_size_singleton_kernel[(num_routes,)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        num_routes,
        BLOCK_SIZE_M=block_size,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


@triton.jit
def _moe_align_block_size_small_grouped_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    NUM_EXPERTS: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
):
    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < NUM_EXPERTS
    counts = tl.zeros((BLOCK_EXPERT,), dtype=tl.int32)

    for route_idx in tl.static_range(0, BLOCK_ROUTES):
        if route_idx < NUM_ROUTES:
            expert_id = tl.load(topk_ids_ptr + route_idx).to(tl.int32)
            counts += tl.where(expert_offsets == expert_id, 1, 0)

    aligned_counts = tl.cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M
    starts = tl.cumsum(aligned_counts, 0) - aligned_counts
    total_tokens = tl.sum(aligned_counts, 0)
    tl.store(num_tokens_post_pad_ptr, total_tokens)

    for block_idx in tl.static_range(0, MAX_BLOCKS_PER_EXPERT):
        block_offset = block_idx * BLOCK_SIZE_M
        valid_block = expert_mask & (block_offset < aligned_counts)
        tl.store(
            expert_ids_ptr + starts // BLOCK_SIZE_M + block_idx,
            expert_offsets,
            mask=valid_block,
        )
        for lane in tl.static_range(0, BLOCK_SIZE_M):
            lane_offset = block_offset + lane
            valid_lane = expert_mask & (lane_offset < aligned_counts)
            tl.store(
                sorted_token_ids_ptr + starts + lane_offset,
                NUM_ROUTES,
                mask=valid_lane,
            )

    ranks = tl.zeros((BLOCK_EXPERT,), dtype=tl.int32)
    for route_idx in tl.static_range(0, BLOCK_ROUTES):
        if route_idx < NUM_ROUTES:
            expert_id = tl.load(topk_ids_ptr + route_idx).to(tl.int32)
            is_expert = expert_offsets == expert_id
            rank = tl.sum(tl.where(is_expert, ranks, 0), 0)
            start = tl.sum(tl.where(is_expert, starts, 0), 0)
            tl.store(sorted_token_ids_ptr + start + rank, route_idx)
            ranks += tl.where(is_expert, 1, 0)


def moe_align_block_size_small_grouped(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    num_routes = topk_ids.numel()
    max_num_tokens_padded = num_routes * block_size
    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty((num_routes,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    _moe_align_block_size_small_grouped_kernel[(1,)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        NUM_EXPERTS=num_experts,
        NUM_ROUTES=num_routes,
        BLOCK_SIZE_M=block_size,
        BLOCK_EXPERT=triton.next_power_of_2(num_experts),
        BLOCK_ROUTES=triton.next_power_of_2(num_routes),
        MAX_BLOCKS_PER_EXPERT=triton.cdiv(num_routes, block_size),
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


@triton.jit
def _moe_align_block_size_compact_count_prefix_init_kernel(
    topk_ids_ptr,
    expert_starts_ptr,
    expert_ranks_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    MAX_PADDED: tl.constexpr,
    INIT_BLOCK: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
):
    route_offsets = tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    route_experts_raw = tl.load(topk_ids_ptr + route_offsets, mask=route_mask, other=-1)
    valid_routes = (
        route_mask & (route_experts_raw >= 0) & (route_experts_raw < NUM_EXPERTS)
    )
    route_experts = tl.where(valid_routes, route_experts_raw, 0).to(tl.int32)
    counts = tl.histogram(route_experts, BLOCK_EXPERT, mask=valid_routes).to(tl.int32)

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < NUM_EXPERTS
    counts = tl.where(expert_mask, counts, 0)
    aligned_counts = tl.cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M
    expert_starts = tl.cumsum(aligned_counts, axis=0) - aligned_counts
    total_tokens = tl.sum(aligned_counts, axis=0)

    # The histogram and prefix sum share one CTA, so counts stay on chip instead
    # of being materialized to global memory between two kernel launches.
    tl.store(expert_starts_ptr + expert_offsets, expert_starts, mask=expert_mask)
    tl.store(expert_ranks_ptr + expert_offsets, 0, mask=expert_mask)
    tl.store(num_tokens_post_pad_ptr, total_tokens)

    # Only the dynamically valid padded region is initialized. Values past
    # total_tokens are never consumed by the grouped GEMM.
    init_offsets = tl.arange(0, INIT_BLOCK)
    for base in tl.static_range(0, MAX_PADDED, INIT_BLOCK):
        offsets = base + init_offsets
        tl.store(
            sorted_token_ids_ptr + offsets,
            NUM_ROUTES,
            mask=offsets < total_tokens,
        )

    # Each active expert normally has a single block for decode-sized routing.
    # The loop also covers skewed routing, up to all routes selecting one expert.
    for block_idx in tl.static_range(0, MAX_BLOCKS_PER_EXPERT):
        valid_block = expert_mask & (block_idx * BLOCK_SIZE_M < aligned_counts)
        output_block = expert_starts // BLOCK_SIZE_M + block_idx
        tl.store(
            expert_ids_ptr + output_block,
            expert_offsets,
            mask=valid_block,
        )


@triton.jit
def _moe_align_block_size_compact_scatter_kernel(
    topk_ids_ptr,
    expert_starts_ptr,
    expert_ranks_ptr,
    sorted_token_ids_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    route_offsets = tl.program_id(0) * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    expert_ids_raw = tl.load(topk_ids_ptr + route_offsets, mask=route_mask, other=-1)
    valid_routes = route_mask & (expert_ids_raw >= 0) & (expert_ids_raw < NUM_EXPERTS)
    expert_ids = tl.where(valid_routes, expert_ids_raw, 0).to(tl.int32)
    ranks = tl.atomic_add(expert_ranks_ptr + expert_ids, 1, mask=valid_routes)
    starts = tl.load(expert_starts_ptr + expert_ids, mask=valid_routes, other=0)
    tl.store(
        sorted_token_ids_ptr + starts + ranks,
        route_offsets,
        mask=valid_routes,
    )


@triton.jit
def _moe_align_block_size_ep_compact_count_prefix_init_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    expert_starts_ptr,
    expert_ranks_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    MAX_PADDED: tl.constexpr,
    INIT_BLOCK: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
):
    """Map global routes and retain only experts owned by this EP rank."""
    route_offsets = tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets, mask=route_mask, other=-1
    )
    valid_global_experts = (
        route_mask
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )
    safe_global_experts = tl.where(valid_global_experts, global_experts_raw, 0).to(
        tl.int64
    )
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global_experts, mask=valid_global_experts, other=-1
    )
    local_route_mask = (
        valid_global_experts
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    safe_local_experts = tl.where(local_route_mask, local_experts_raw, 0).to(tl.int32)
    counts = tl.histogram(safe_local_experts, BLOCK_EXPERT, mask=local_route_mask).to(
        tl.int32
    )

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < NUM_LOCAL_EXPERTS
    counts = tl.where(expert_mask, counts, 0)
    aligned_counts = tl.cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M
    expert_starts = tl.cumsum(aligned_counts, axis=0) - aligned_counts
    total_tokens = tl.sum(aligned_counts, axis=0)

    tl.store(expert_starts_ptr + expert_offsets, expert_starts, mask=expert_mask)
    tl.store(expert_ranks_ptr + expert_offsets, 0, mask=expert_mask)
    tl.store(num_tokens_post_pad_ptr, total_tokens)

    init_offsets = tl.arange(0, INIT_BLOCK)
    for base in tl.static_range(0, MAX_PADDED, INIT_BLOCK):
        offsets = base + init_offsets
        tl.store(
            sorted_token_ids_ptr + offsets,
            NUM_ROUTES,
            mask=offsets < total_tokens,
        )

    for block_idx in tl.static_range(0, MAX_BLOCKS_PER_EXPERT):
        valid_block = expert_mask & (block_idx * BLOCK_SIZE_M < aligned_counts)
        output_block = expert_starts // BLOCK_SIZE_M + block_idx
        tl.store(
            expert_ids_ptr + output_block,
            expert_offsets,
            mask=valid_block,
        )


@triton.jit
def _moe_align_block_size_ep_compact_scatter_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    expert_starts_ptr,
    expert_ranks_ptr,
    sorted_token_ids_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    route_offsets = tl.program_id(0) * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets, mask=route_mask, other=-1
    )
    valid_global_experts = (
        route_mask
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )
    safe_global_experts = tl.where(valid_global_experts, global_experts_raw, 0).to(
        tl.int64
    )
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global_experts, mask=valid_global_experts, other=-1
    )
    local_route_mask = (
        valid_global_experts
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    safe_local_experts = tl.where(local_route_mask, local_experts_raw, 0).to(tl.int32)
    ranks = tl.atomic_add(
        expert_ranks_ptr + safe_local_experts, 1, mask=local_route_mask
    )
    starts = tl.load(
        expert_starts_ptr + safe_local_experts,
        mask=local_route_mask,
        other=0,
    )
    tl.store(
        sorted_token_ids_ptr + starts + ranks,
        route_offsets,
        mask=local_route_mask,
    )


@triton.jit
def _moe_align_block_size_ep_route_block_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    """Emit one GEMM block for each route owned by this EP rank."""
    route_offsets = tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets, mask=route_mask, other=-1
    )
    valid_global_experts = (
        route_mask
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )

    # Never use an untrusted global ID for pointer arithmetic. Keeping this
    # index in int64 also prevents large int64 route IDs from wrapping to a
    # seemingly valid int32 expert before the bounds check.
    safe_global_experts = tl.where(valid_global_experts, global_experts_raw, 0).to(
        tl.int64
    )
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global_experts, mask=valid_global_experts, other=-1
    )
    local_route_mask = (
        valid_global_experts
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )

    # The inclusive prefix sum gives each local route a deterministic compact
    # block rank. Repeated experts deliberately remain separate blocks: GEMM
    # correctness only requires every block to contain routes for one expert.
    route_ranks = tl.cumsum(local_route_mask.to(tl.int32), axis=0) - 1
    num_local_routes = tl.sum(local_route_mask.to(tl.int32), axis=0)
    tl.store(num_tokens_post_pad_ptr, num_local_routes * BLOCK_SIZE_M)
    tl.store(
        expert_ids_ptr + route_ranks,
        local_experts_raw.to(tl.int32),
        mask=local_route_mask,
    )

    block_lanes = tl.arange(0, BLOCK_SIZE_M)
    output_offsets = route_ranks[:, None] * BLOCK_SIZE_M + block_lanes[None, :]
    output_values = tl.where(
        block_lanes[None, :] == 0,
        route_offsets[:, None],
        NUM_ROUTES,
    )
    tl.store(
        sorted_token_ids_ptr + output_offsets,
        output_values,
        mask=local_route_mask[:, None],
    )


def moe_align_block_size_compact(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Align a decode-sized routing table without an E x E count matrix."""
    num_routes = topk_ids.numel()
    max_num_tokens_padded = num_routes + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)

    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    expert_starts = torch.empty(
        (num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ranks = torch.empty_like(expert_starts)

    block_expert = triton.next_power_of_2(num_experts)
    _moe_align_block_size_compact_count_prefix_init_kernel[(1,)](
        topk_ids,
        expert_starts,
        expert_ranks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        BLOCK_SIZE_M=block_size,
        BLOCK_EXPERT=block_expert,
        BLOCK_ROUTES=triton.next_power_of_2(num_routes),
        MAX_PADDED=max_num_tokens_padded,
        INIT_BLOCK=256,
        MAX_BLOCKS_PER_EXPERT=triton.cdiv(num_routes, block_size),
        num_warps=4,
    )
    scatter_block = 128
    _moe_align_block_size_compact_scatter_kernel[
        (triton.cdiv(num_routes, scatter_block),)
    ](
        topk_ids,
        expert_starts,
        expert_ranks,
        sorted_token_ids,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        BLOCK_ROUTES=scatter_block,
        num_warps=4,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size_ep_compact(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    block_size: int,
    local_num_experts: int,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Compact alignment that drops routes owned by other EP ranks."""
    num_routes = topk_ids.numel()
    max_num_tokens_padded = num_routes + local_num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)

    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    expert_starts = torch.empty(
        (local_num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ranks = torch.empty_like(expert_starts)

    block_expert = triton.next_power_of_2(local_num_experts)
    _moe_align_block_size_ep_compact_count_prefix_init_kernel[(1,)](
        topk_ids,
        expert_map,
        expert_starts,
        expert_ranks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        NUM_ROUTES=num_routes,
        NUM_GLOBAL_EXPERTS=expert_map.numel(),
        NUM_LOCAL_EXPERTS=local_num_experts,
        BLOCK_SIZE_M=block_size,
        BLOCK_EXPERT=block_expert,
        BLOCK_ROUTES=triton.next_power_of_2(num_routes),
        MAX_PADDED=max_num_tokens_padded,
        INIT_BLOCK=256,
        MAX_BLOCKS_PER_EXPERT=triton.cdiv(num_routes, block_size),
        num_warps=4,
    )
    scatter_block = 128
    _moe_align_block_size_ep_compact_scatter_kernel[
        (triton.cdiv(num_routes, scatter_block),)
    ](
        topk_ids,
        expert_map,
        expert_starts,
        expert_ranks,
        sorted_token_ids,
        NUM_ROUTES=num_routes,
        NUM_GLOBAL_EXPERTS=expert_map.numel(),
        NUM_LOCAL_EXPERTS=local_num_experts,
        BLOCK_ROUTES=scatter_block,
        num_warps=4,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size_ep_route_block(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    block_size: int,
    local_num_experts: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Create one padded BM block per valid local EP route in one kernel.

    The route order is stable, but routes selecting the same expert are not
    grouped into a shared block. This layout is intended for very small decode
    routing tables where avoiding a second alignment launch is more valuable
    than packing multiple routes for one expert into the same GEMM block.
    """
    if block_size != EP_ROUTE_BLOCK_SIZE:
        raise ValueError(
            "EP route-block alignment requires "
            f"block_size={EP_ROUTE_BLOCK_SIZE}, got {block_size}"
        )
    if local_num_experts <= 0:
        raise ValueError(f"local_num_experts must be positive, got {local_num_experts}")
    if not topk_ids.is_cuda:
        raise ValueError("EP route-block alignment requires CUDA tensors")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must have int32 or int64 dtype")
    if expert_map.ndim != 1 or expert_map.numel() <= 0:
        raise ValueError("expert_map must be a non-empty 1D tensor")
    if not expert_map.is_contiguous():
        raise ValueError("expert_map must be contiguous")
    if expert_map.device != topk_ids.device:
        raise ValueError("expert_map and topk_ids must be on the same device")
    if expert_map.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_map must have int32 or int64 dtype")

    num_routes = topk_ids.numel()
    if not 0 < num_routes <= EP_ROUTE_BLOCK_MAX_ROUTES:
        raise ValueError(
            "EP route-block alignment supports between 1 and "
            f"{EP_ROUTE_BLOCK_MAX_ROUTES} routes, got {num_routes}"
        )

    # The largest possible dynamic valid region occurs when every route is
    # local. Each block is initialized completely by the single kernel.
    sorted_token_ids = torch.empty(
        (num_routes * block_size,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty((num_routes,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    _moe_align_block_size_ep_route_block_kernel[(1,)](
        topk_ids,
        expert_map,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        NUM_ROUTES=num_routes,
        NUM_GLOBAL_EXPERTS=expert_map.numel(),
        NUM_LOCAL_EXPERTS=local_num_experts,
        BLOCK_SIZE_M=block_size,
        BLOCK_ROUTES=triton.next_power_of_2(num_routes),
        num_warps=4,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    local_num_experts: Optional[int] = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must have int32 or int64 dtype")
    if expert_map is not None:
        if expert_map.ndim != 1 or expert_map.numel() != num_experts:
            raise ValueError(
                f"expert_map must have shape ({num_experts},), "
                f"got {tuple(expert_map.shape)}"
            )
        if not expert_map.is_contiguous():
            raise ValueError("expert_map must be contiguous")
        if expert_map.device != topk_ids.device:
            raise ValueError("expert_map and topk_ids must be on the same device")
        if expert_map.dtype not in (torch.int32, torch.int64):
            raise ValueError("expert_map must have int32 or int64 dtype")
    num_routes = topk_ids.numel()
    if (
        expert_map is not None
        and ignore_invalid_experts
        and local_num_experts == EP_COMPACT_LOCAL_EXPERTS
        and num_experts == EP_COMPACT_GLOBAL_EXPERTS
        and topk_ids.is_cuda
        and _supports_compact_align(topk_ids.device.index)
        and 0 < num_routes <= EP_COMPACT_MAX_DECODE_ROUTES
    ):
        return moe_align_block_size_ep_compact(
            topk_ids,
            expert_map,
            block_size,
            local_num_experts,
            pad_sorted_ids,
        )
    if (
        expert_map is None
        and topk_ids.is_cuda
        # On H20 at the Qwen4 decode routing point, compact is 7.07 us versus
        # 10.12 us for TLE cluster and 26.71 us for TLE cooperative atomic.
        # Keep this override exact; other TLE workloads remain unmodified.
        and (
            not HAS_TLE
            or (
                num_routes == COMPACT_ALIGN_TLE_OVERRIDE_ROUTES
                and num_experts == COMPACT_ALIGN_TLE_OVERRIDE_EXPERTS
            )
        )
        # The compact launch parameters are measured on Hopper. Keep older
        # architectures on their existing path until they are benchmarked.
        and _supports_compact_align(topk_ids.device.index)
        and block_size == COMPACT_ALIGN_BLOCK_SIZE
        and 0 < num_routes <= COMPACT_ALIGN_MAX_ROUTES
        and COMPACT_ALIGN_MIN_EXPERTS <= num_experts <= COMPACT_ALIGN_MAX_EXPERTS
    ):
        return moe_align_block_size_compact(
            topk_ids,
            block_size,
            num_experts,
            pad_sorted_ids,
        )

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad
