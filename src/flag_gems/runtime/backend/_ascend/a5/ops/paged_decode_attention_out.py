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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

_LOG2E = 1.4426950408889634


@triton.jit
def _kv_page_partial(
    q_ptr,
    k_ptr,
    v_ptr,
    bt_ptr,
    mask_ptr,
    part_ptr,
    pages: tl.constexpr,
    q_s_h: tl.constexpr,
    q_s_d: tl.constexpr,
    k_s_p: tl.constexpr,
    k_s_t: tl.constexpr,
    k_s_d: tl.constexpr,
    v_s_p: tl.constexpr,
    v_s_t: tl.constexpr,
    v_s_d: tl.constexpr,
    bt_s_0: tl.constexpr,
    bt_s_1: tl.constexpr,
    SCALE_LOG2E: tl.constexpr,
    L_KV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M_PAD: tl.constexpr,
    HEADS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)

    # All HEADS query rows in one cube tile; rows >= HEADS stay zero.
    hmask = (offs_m < HEADS)[:, None]
    q_tile = tl.load(
        q_ptr + offs_m[:, None] * q_s_h + offs_d[None, :] * q_s_d, mask=hmask, other=0.0
    )

    bid = tl.load(bt_ptr + 0 * bt_s_0 + pid * bt_s_1).to(tl.int64)
    valid_page = (bid >= 0) & (bid < pages)
    # K is loaded already transposed ([D, BLOCK_N]) so the cube consumes it directly:
    # this removes the 64KiB in-UB bf16 transpose. Operand values are identical,
    # so the dot result is unchanged.
    k_t = tl.load(
        k_ptr + bid * k_s_p + offs_d[:, None] * k_s_d + offs_n[None, :] * k_s_t,
        mask=valid_page,
        other=0.0,
    )
    v_tile = tl.load(
        v_ptr + bid * v_s_p + offs_n[:, None] * v_s_t + offs_d[None, :] * v_s_d,
        mask=valid_page,
        other=0.0,
    )

    # Raw (unscaled) scores; the 0.0625 attention scale lives in the exponent.
    s = tl.dot(q_tile, k_t, out_dtype=tl.float32)
    j = pid * BLOCK_N + offs_n
    visible = (j < L_KV) & valid_page
    # A single decode query is at absolute position L_KV-1. This
    # specialization accepts the standard 2048-square causal template.
    masked = tl.load(mask_ptr + (L_KV - 1) * 2048 + j, mask=visible, other=1)
    visible = visible & (masked == 0)
    logits = tl.where(visible[None, :], s * SCALE_LOG2E, -float("inf"))
    maximum = tl.max(logits, 1)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    prob = tl.exp2(logits - safe[:, None])
    den = tl.sum(prob, axis=1)
    acc = tl.dot(prob.to(tl.bfloat16), v_tile, out_dtype=tl.float32)
    stride = HEADS * (D + 2)
    base = part_ptr + pid * stride
    tl.store(base + offs_m[:, None] * D + offs_d[None, :], acc, mask=hmask)
    tl.store(base + HEADS * D + offs_m, den, mask=offs_m < HEADS)
    tl.store(base + HEADS * (D + 1) + offs_m, maximum, mask=offs_m < HEADS)


@triton.jit
def _merge_pages(
    part,
    out,
    count: tl.constexpr,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    HEADS: tl.constexpr,
):
    head = tl.program_id(0)
    p = tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    base = part + p * (HEADS * (D + 2))
    maximum = tl.load(
        base + HEADS * (D + 1) + head, mask=p < count, other=-float("inf")
    )
    max_all = tl.max(maximum, 0)
    max_all = tl.where(max_all == -float("inf"), 0.0, max_all)
    factor = tl.exp2(maximum - max_all)
    den = tl.sum(
        tl.load(base + HEADS * D + head, mask=p < count, other=0.0) * factor, axis=0
    )
    vec = tl.load(
        base[:, None] + head * D + d[None, :], mask=(p < count)[:, None], other=0.0
    )
    result = tl.sum(vec * factor[:, None], axis=0) / tl.where(den > 0, den, 1.0)
    tl.store(out + head * D + d, result.to(tl.bfloat16))


def _launch(
    query, key, value, atten_mask, block_table, actual_seq_lengths_kv, workspace, out
):
    length = actual_seq_lengths_kv[0]
    count = triton.cdiv(length, 128)
    partial = workspace.view(torch.float32)
    _kv_page_partial[(count,)](
        query,
        key,
        value,
        block_table,
        atten_mask,
        partial,
        key.shape[0],
        query.stride(1),
        query.stride(2),
        *key.stride(),
        *value.stride(),
        *block_table.stride(),
        SCALE_LOG2E=0.0625 * _LOG2E,
        L_KV=length,
        D=256,
        BLOCK_N=128,
        M_PAD=16,
        HEADS=8,
    )
    _merge_pages[(8,)](
        partial,
        out[0],
        count=count,
        BLOCK=triton.next_power_of_2(count),
        D=256,
        HEADS=8,
    )
    return tuple(out)


def paged_decode_attention_out(
    query, key, value, atten_mask, block_table, actual_seq_lengths_kv, workspace, out
):
    """Single-query BF16 paged causal decode for eight 256-channel query heads and one KV head."""
    if (
        query.shape != (1, 8, 256)
        or key.ndim != 3
        or key.shape[1:] != (128, 256)
        or value.shape != key.shape
        or not isinstance(actual_seq_lengths_kv, (tuple, list))
        or len(actual_seq_lengths_kv) != 1
        or not isinstance(actual_seq_lengths_kv[0], int)
    ):
        raise ValueError("expected a single query, one KV head and 128-token pages")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.dtype != torch.int32
        or block_table.stride(-1) != 1
        or not 0 < actual_seq_lengths_kv[0] <= block_table.shape[1] * 128
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != key.dtype
        or atten_mask.shape != (2048, 2048)
        or atten_mask.dtype != torch.int8
    ):
        raise ValueError(
            "invalid block table, sequence length, causal mask or KV dtype"
        )
    length = actual_seq_lengths_kv[0]
    required_bytes = triton.cdiv(length, 128) * 8 * (256 + 2) * 4
    if length > 2048 or not atten_mask.is_contiguous() or key.shape[0] == 0:
        raise ValueError(
            "expected at most 2048 KV positions and a contiguous causal template"
        )
    if (
        not isinstance(out, (list, tuple))
        or len(out) != 2
        or out[0].shape != query.shape
        or out[0].dtype != query.dtype
        or not out[0].is_contiguous()
        or out[1].numel() != 1
        or out[1].dtype != query.dtype
        or workspace.ndim != 1
        or workspace.numel() * workspace.element_size() < required_bytes
        or not workspace.is_contiguous()
        or workspace.dtype != torch.bfloat16
        or workspace.numel() % 2
        or workspace.storage_offset() % 2
        or any(
            t.device != query.device for t in (atten_mask, block_table, workspace, *out)
        )
    ):
        raise ValueError("invalid output or workspace")
    # Workspace and outputs are written by different CTAs. Reject overlapping
    # storage, including aliases of a read-only input.
    written = [workspace, *out]
    readonly = [query, key, value, atten_mask, block_table]
    for i, t in enumerate(written):
        if any(torch._C._is_alias_of(t, other) for other in readonly + written[:i]):
            raise ValueError("workspace and output storage must be independent")
    if (
        query.device.type != "npu"
        or key.device != query.device
        or value.device != query.device
    ):
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(query.device):
        return _launch(
            query,
            key,
            value,
            atten_mask,
            block_table,
            actual_seq_lengths_kv,
            workspace,
            out,
        )
