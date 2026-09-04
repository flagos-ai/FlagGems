#!/usr/bin/env python3
"""Full-op prototype for deterministic local-route popcount fast paths.

This file deliberately keeps every experimental kernel outside production
modules.  It monkeypatches the Python dispatch only while constructing eager
calls or CUDA graphs:

* compact EP alignment emits one local-route bit mask per token;
* GEMM2 writes a token with exactly one local route directly to final output;
* GEMM2 retains cache3 only for tokens with multiple local routes;
* combine writes zero for no-local tokens, skips unique tokens, and reduces
  collisions in the original top-k order.

When final output overlaps cache2, the wrapper falls back to the production
GEMM2 + moe_sum_ep path because direct output would race GEMM2 cache2 reads.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import statistics
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _ep_compact_count_prefix_mask_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    expert_starts_ptr,
    expert_ranks_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    local_route_masks_ptr,
    NUM_TOKENS: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    MAX_PADDED: tl.constexpr,
    INIT_BLOCK: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
):
    route_offsets = tl.arange(0, BLOCK_ROUTES)
    route_in_bounds = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets, mask=route_in_bounds, other=-1
    )
    valid_global_experts = (
        route_in_bounds
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )
    safe_global_experts = tl.where(valid_global_experts, global_experts_raw, 0).to(
        tl.int64
    )
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global_experts,
        mask=valid_global_experts,
        other=-1,
    )
    local_route = (
        valid_global_experts
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    safe_local_experts = tl.where(local_route, local_experts_raw, 0).to(tl.int32)
    counts = tl.histogram(safe_local_experts, BLOCK_EXPERT, mask=local_route).to(
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

    # Routes are token-major and TOP_K=8.  Reshape the existing route mapping
    # result and emit one bit mask without another topk_ids/expert_map pass.
    route_bits = tl.where(
        local_route,
        1 << (route_offsets % TOP_K),
        0,
    ).to(tl.int32)
    route_bits = tl.reshape(route_bits, (BLOCK_TOKENS, TOP_K))
    token_masks = tl.sum(route_bits, axis=1)
    token_offsets = tl.arange(0, BLOCK_TOKENS)
    tl.store(
        local_route_masks_ptr + token_offsets,
        token_masks,
        mask=token_offsets < NUM_TOKENS,
    )
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
def _gemm2_popcount_kernel(
    a_ptr,
    b_ptr,
    cache3_ptr,
    output_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    local_route_masks_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs_m).to(
        tl.int64
    )
    token_mask = offs_token < NUM_ROUTES
    safe_token = tl.where(token_mask, offs_token, 0)
    token_idx = safe_token // TOP_K
    route_masks = tl.load(local_route_masks_ptr + token_idx, mask=token_mask, other=0)
    single_route = (route_masks != 0) & ((route_masks & (route_masks - 1)) == 0)

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + safe_token[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    route_weight = tl.load(
        topk_weights_ptr + safe_token,
        mask=token_mask,
        other=0.0,
    )
    acc *= route_weight[:, None]
    result = acc.to(tl.bfloat16)
    hidden_mask = offs_n[None, :] < N

    output_ptrs = (
        output_ptr + token_idx[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    tl.store(
        output_ptrs,
        result,
        mask=token_mask[:, None] & single_route[:, None] & hidden_mask,
    )
    cache_ptrs = (
        cache3_ptr + safe_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    )
    tl.store(
        cache_ptrs,
        result,
        mask=token_mask[:, None] & ~single_route[:, None] & hidden_mask,
    )


@triton.jit
def _collision_combine_kernel(
    cache3_ptr,
    output_ptr,
    local_route_masks_ptr,
    NUM_TOKENS: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_topk: tl.constexpr,
    input_stride_hidden: tl.constexpr,
    output_stride_token: tl.constexpr,
    output_stride_hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    if token_idx >= NUM_TOKENS:
        return
    route_mask = tl.load(local_route_masks_ptr + token_idx)
    single_route = route_mask != 0 and (route_mask & (route_mask - 1)) == 0
    if single_route:
        return

    output_ptrs = (
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden
    )
    if route_mask == 0:
        tl.store(output_ptrs, 0.0, mask=hidden_mask)
        return

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = cache3_ptr + token_idx * input_stride_token
    for route_idx in tl.static_range(TOP_K):
        is_local = (route_mask & (1 << route_idx)) != 0
        route_ptr = input_base + route_idx * input_stride_topk
        route_data = tl.load(
            route_ptr + hidden_offsets * input_stride_hidden,
            mask=hidden_mask & is_local,
            other=0.0,
        )
        acc += route_data
    tl.store(output_ptrs, acc, mask=hidden_mask)


@dataclass
class PrototypeState:
    enabled: bool = False
    direct_used: bool = False
    output: torch.Tensor | None = None
    local_route_masks: torch.Tensor | None = None


class PopcountPrototype:
    def __init__(self, fm, align_module):
        self.fm = fm
        self.align_module = align_module
        self.state = PrototypeState()
        self.original_align = fm.moe_align_block_size
        self.original_dispatch = fm.dispatch_fused_moe_kernel
        self.original_sum = fm.moe_sum_ep
        self.dispatch_signature = inspect.signature(self.original_dispatch)

    def install(self):
        self.fm.moe_align_block_size = self.align
        self.fm.dispatch_fused_moe_kernel = self.dispatch
        self.fm.moe_sum_ep = self.sum

    def restore(self):
        self.fm.moe_align_block_size = self.original_align
        self.fm.dispatch_fused_moe_kernel = self.original_dispatch
        self.fm.moe_sum_ep = self.original_sum

    def align(
        self,
        topk_ids,
        block_size,
        num_experts,
        expert_map=None,
        pad_sorted_ids=False,
        ignore_invalid_experts=False,
        *,
        local_num_experts=None,
    ):
        if not (
            self.state.enabled
            and expert_map is not None
            and ignore_invalid_experts
            and local_num_experts == 18
            and num_experts == 288
            and block_size == 16
            and topk_ids.ndim == 2
            and topk_ids.shape[1] == 8
            and 0 < topk_ids.numel() <= 1024
        ):
            return self.original_align(
                topk_ids,
                block_size,
                num_experts,
                expert_map,
                pad_sorted_ids,
                ignore_invalid_experts,
                local_num_experts=local_num_experts,
            )

        num_tokens, topk = topk_ids.shape
        num_routes = topk_ids.numel()
        max_padded = num_routes + local_num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_padded = self.align_module.round_up(max_padded, block_size)
        sorted_ids = torch.empty(max_padded, device=topk_ids.device, dtype=torch.int32)
        expert_ids = torch.empty(
            triton.cdiv(max_padded, block_size),
            device=topk_ids.device,
            dtype=torch.int32,
        )
        num_post_pad = torch.empty(1, device=topk_ids.device, dtype=torch.int32)
        expert_starts = torch.empty(
            local_num_experts, device=topk_ids.device, dtype=torch.int32
        )
        expert_ranks = torch.empty_like(expert_starts)
        local_route_masks = torch.empty(
            num_tokens, device=topk_ids.device, dtype=torch.int32
        )
        block_routes = triton.next_power_of_2(num_routes)
        _ep_compact_count_prefix_mask_kernel[(1,)](
            topk_ids,
            expert_map,
            expert_starts,
            expert_ranks,
            sorted_ids,
            expert_ids,
            num_post_pad,
            local_route_masks,
            NUM_TOKENS=num_tokens,
            TOP_K=topk,
            NUM_ROUTES=num_routes,
            NUM_GLOBAL_EXPERTS=expert_map.numel(),
            NUM_LOCAL_EXPERTS=local_num_experts,
            BLOCK_SIZE_M=block_size,
            BLOCK_EXPERT=triton.next_power_of_2(local_num_experts),
            BLOCK_ROUTES=block_routes,
            BLOCK_TOKENS=block_routes // topk,
            MAX_PADDED=max_padded,
            INIT_BLOCK=256,
            MAX_BLOCKS_PER_EXPERT=triton.cdiv(num_routes, block_size),
            num_warps=4,
        )
        scatter_block = 128
        self.align_module._moe_align_block_size_ep_compact_scatter_kernel[
            (triton.cdiv(num_routes, scatter_block),)
        ](
            topk_ids,
            expert_map,
            expert_starts,
            expert_ranks,
            sorted_ids,
            NUM_ROUTES=num_routes,
            NUM_GLOBAL_EXPERTS=expert_map.numel(),
            NUM_LOCAL_EXPERTS=local_num_experts,
            BLOCK_ROUTES=scatter_block,
            num_warps=4,
        )
        self.state.local_route_masks = local_route_masks
        return sorted_ids, expert_ids, num_post_pad

    def dispatch(self, *args, **kwargs):
        bound = self.dispatch_signature.bind_partial(*args, **kwargs).arguments
        A, B, C = bound["A"], bound["B"], bound["C"]
        config = bound["config"]
        is_ep_gemm2 = (
            self.state.enabled
            and self.state.output is not None
            and self.state.local_route_masks is not None
            and A.dtype == torch.bfloat16
            and B.dtype == torch.bfloat16
            and A.ndim == 2
            and A.shape[1] == 2048
            and tuple(B.shape[1:]) == (4096, 2048)
            and tuple(C.shape[-2:]) == (8, 4096)
            and bound["top_k"] == 1
            and bound["mul_routed_weight"]
            and not bound.get("FUSE_SILU", False)
            and not any(
                bound[name]
                for name in (
                    "use_fp8_w8a8",
                    "use_int8_w8a8",
                    "use_int8_w8a16",
                    "use_int4_w4a16",
                )
            )
            and bound.get("B_bias") is None
            and config["BLOCK_SIZE_M"] == 16
            and config["BLOCK_SIZE_N"] == 128
            and config["BLOCK_SIZE_K"] == 64
        )
        if not is_ep_gemm2:
            return self.original_dispatch(*args, **kwargs)

        if self.fm._tensors_overlap(self.state.output, A):
            self.state.direct_used = False
            return self.original_dispatch(*args, **kwargs)

        sorted_ids = bound["sorted_token_ids"]
        expert_ids = bound["expert_ids"]
        num_post_pad = bound["num_tokens_post_padded"]
        topk_weights = bound["topk_weights"]
        em = sorted_ids.numel()
        grid = (
            triton.cdiv(em, config["BLOCK_SIZE_M"])
            * triton.cdiv(B.shape[1], config["BLOCK_SIZE_N"]),
        )
        _gemm2_popcount_kernel[grid](
            A,
            B,
            C,
            self.state.output,
            topk_weights,
            sorted_ids,
            expert_ids,
            num_post_pad,
            self.state.local_route_masks,
            N=B.shape[1],
            K=B.shape[2],
            EM=em,
            NUM_ROUTES=A.shape[0],
            TOP_K=8,
            stride_am=A.stride(0),
            stride_ak=A.stride(1),
            stride_be=B.stride(0),
            stride_bk=B.stride(2),
            stride_bn=B.stride(1),
            stride_cm=C.stride(1),
            stride_cn=C.stride(2),
            stride_om=self.state.output.stride(0),
            stride_on=self.state.output.stride(1),
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        self.state.direct_used = True

    def sum(
        self,
        input,
        output,
        topk_ids,
        expert_map,
        local_num_experts,
        *,
        fixed_block_size=None,
        fixed_num_warps=None,
    ):
        if not self.state.direct_used:
            return self.original_sum(
                input,
                output,
                topk_ids,
                expert_map,
                local_num_experts,
                fixed_block_size=fixed_block_size,
                fixed_num_warps=fixed_num_warps,
            )
        masks = self.state.local_route_masks
        m, topk, hidden = input.shape
        block_size = 256
        _collision_combine_kernel[(m, triton.cdiv(hidden, block_size))](
            input,
            output,
            masks,
            NUM_TOKENS=m,
            TOP_K=topk,
            HIDDEN_SIZE=hidden,
            input_stride_token=input.stride(0),
            input_stride_topk=input.stride(1),
            input_stride_hidden=input.stride(2),
            output_stride_token=output.stride(0),
            output_stride_hidden=output.stride(1),
            BLOCK_SIZE=block_size,
            num_warps=1,
        )


def capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            result = fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    return graph, result


def time_graph(graph, rounds, replays):
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end) / replays))
    return samples


def make_ids(routing, m, topk, global_e, local_e, device, dtype):
    if routing == "uniform":
        ids = torch.stack(
            [torch.randperm(global_e, device=device)[:topk] for _ in range(m)]
        )
    elif routing == "no-local":
        ids = torch.stack(
            [
                torch.randperm(global_e - local_e, device=device)[:topk] + local_e
                for _ in range(m)
            ]
        )
    elif routing == "local4":
        rows = []
        for _ in range(m):
            local = torch.randperm(local_e, device=device)[:4]
            remote = (
                torch.randperm(global_e - local_e, device=device)[: topk - 4] + local_e
            )
            rows.append(torch.cat((local, remote)))
        ids = torch.stack(rows)
    elif routing == "all-local":
        ids = torch.stack(
            [torch.randperm(local_e, device=device)[:topk] for _ in range(m)]
        )
    elif routing == "invalid":
        ids = torch.stack(
            [torch.randperm(global_e, device=device)[:topk] for _ in range(m)]
        )
        ids[0, 0] = -1
        ids[1, 1] = global_e + 7
    else:
        raise ValueError(routing)
    return ids.to(dtype)


def make_buffers(m, topk, h, intermediate, device, dtype, *, alias_output):
    cache13 = torch.empty(
        m * topk * max(2 * intermediate, h), device=device, dtype=dtype
    )
    cache2 = torch.empty(m * topk * intermediate, device=device, dtype=dtype)
    output = (
        cache2[: m * h].view(m, h)
        if alias_output
        else torch.empty((m, h), device=device, dtype=dtype)
    )
    return {"cache13": cache13, "cache2": cache2, "output": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routing",
        choices=("uniform", "no-local", "local4", "all-local", "invalid"),
        default="uniform",
    )
    parser.add_argument("--ids-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument("--alias-output", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()

    fm = importlib.import_module("flag_gems.fused.fused_moe")
    align_module = importlib.import_module("flag_gems.fused.moe_align_block_size")
    prototype = PopcountPrototype(fm, align_module)
    prototype.install()
    try:
        torch.manual_seed(20260824)
        m, global_e, local_e, h, intermediate, topk = 96, 288, 18, 4096, 2048, 8
        device, dtype = torch.device("cuda"), torch.bfloat16
        ids_dtype = torch.int32 if args.ids_dtype == "int32" else torch.int64
        hidden = torch.randn((m, h), device=device, dtype=dtype)
        w1 = torch.empty((local_e, 2 * intermediate, h), device=device, dtype=dtype)
        w1.normal_(std=h**-0.5)
        w2 = torch.empty((local_e, h, intermediate), device=device, dtype=dtype)
        w2.normal_(std=intermediate**-0.5)
        ids = make_ids(
            args.routing, m, topk, global_e, local_e, device, ids_dtype
        ).contiguous()
        weights = torch.rand((m, topk), device=device, dtype=torch.float32)
        weights = (weights / weights.sum(-1, keepdim=True)).to(dtype).contiguous()
        expert_map = torch.full((global_e,), -1, device=device, dtype=torch.int32)
        expert_map[:local_e] = torch.arange(local_e, device=device, dtype=torch.int32)
        buffers = [
            make_buffers(
                m,
                topk,
                h,
                intermediate,
                device,
                dtype,
                alias_output=args.alias_output,
            )
            for _ in range(2)
        ]

        def op(policy):
            prototype.state.enabled = bool(policy)
            prototype.state.direct_used = False
            prototype.state.output = buffers[policy]["output"]
            prototype.state.local_route_masks = None
            return fm.fused_experts_impl(
                hidden,
                w1,
                w2,
                weights,
                ids,
                global_num_experts=global_e,
                expert_map=expert_map,
                gemm1_clamp_limit=10.0,
                output=buffers[policy]["output"],
                intermediate_cache13=buffers[policy]["cache13"],
                intermediate_cache2=buffers[policy]["cache2"],
            )

        eager = [op(0).clone(), op(1).clone()]
        torch.cuda.synchronize()
        graphs, graph_outputs = [], []
        for policy in (0, 1):
            graph, graph_output = capture(lambda selected=policy: op(selected))
            graphs.append(graph)
            graph_outputs.append(graph_output)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        initial_baseline_graph_bitwise = bool(torch.equal(eager[0], graph_outputs[0]))
        initial_popcount_graph_bitwise = bool(torch.equal(eager[1], graph_outputs[1]))

        samples = [[], []]
        for round_idx in range(args.rounds):
            order = (0, 1) if round_idx % 2 == 0 else (1, 0)
            for policy in order:
                samples[policy].extend(time_graph(graphs[policy], 1, args.replays))
        medians = [statistics.median(values) for values in samples]

        # Replay both graphs with new routing/weights to prove metadata is not
        # frozen at capture time.
        updated_ids = make_ids(
            "invalid" if args.routing != "invalid" else "uniform",
            m,
            topk,
            global_e,
            local_e,
            device,
            ids_dtype,
        )
        updated_weights = torch.rand_like(weights.float())
        updated_weights = (updated_weights / updated_weights.sum(-1, keepdim=True)).to(
            dtype
        )
        ids.copy_(updated_ids)
        weights.copy_(updated_weights)
        graphs[0].replay()
        graphs[1].replay()
        torch.cuda.synchronize()
        updated_graph_bitwise = bool(torch.equal(graph_outputs[0], graph_outputs[1]))

        mapped = torch.full_like(ids, -1, dtype=torch.int32)
        valid = (ids >= 0) & (ids < global_e)
        mapped[valid] = expert_map[ids[valid].long()]
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "routing": args.routing,
                    "ids_dtype": args.ids_dtype,
                    "alias_output": args.alias_output,
                    "local_routes_after_update": int((mapped >= 0).sum().item()),
                    "baseline_ms": {"median": medians[0], "samples": samples[0]},
                    "popcount_ms": {"median": medians[1], "samples": samples[1]},
                    "delta_percent": 100 * (medians[1] - medians[0]) / medians[0],
                    "eager_bitwise": bool(torch.equal(eager[0], eager[1])),
                    "baseline_graph_bitwise": initial_baseline_graph_bitwise,
                    "popcount_graph_bitwise": initial_popcount_graph_bitwise,
                    "updated_graph_bitwise": updated_graph_bitwise,
                },
                indent=2,
            )
        )
    finally:
        prototype.restore()


if __name__ == "__main__":
    main()
