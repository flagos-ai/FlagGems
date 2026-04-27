import logging
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, libtuner


def _triton_version_at_least(major: int, minor: int, patch: int = 0) -> bool:
    version = str(getattr(triton, "__version__", "0.0.0")).split("+", 1)[0]
    parts = version.split(".")
    parsed = []
    for part in parts[:3]:
        digits = []
        for ch in part:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        parsed.append(int("".join(digits)) if digits else 0)
    while len(parsed) < 3:
        parsed.append(0)
    return tuple(parsed) >= (major, minor, patch)


if _triton_version_at_least(3, 6, 0):
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
        tl.store(expert_ids_ptr + offs, 0, mask=offs < numel_expert_ids)
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
        tl.store(expert_ids_ptr + offs, 0, mask=mask)

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
    tl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

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


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
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
