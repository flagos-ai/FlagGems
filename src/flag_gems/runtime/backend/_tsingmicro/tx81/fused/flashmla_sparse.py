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

import functools
import inspect
import logging
import os
import time

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)

# Sparse MLA on the tsingmicro (txda) backend.
#
# The indexer (kpool=4) emits, per query, pools of 4 consecutive tokens; after
# paged block mapping (block_size=64, 4|64) and the COMPACT_TO_FRONT physical
# conversion, the *physical* topk is a list of scattered token ids whose only
# structure is that groups of 4 are contiguous rows.  K and V are the same
# tensor (the MLA latent kv_c), so one gather feeds both.
#
# txda lowering constraints (established by minimal-kernel experiments):
#   * `tl.trans`/`tl.permute` of a *gathered* load is unsafe -- a 3-D
#     `tl.permute` of a dynamically-indexed load lowers via
#     `GetStructuredStateOp`, which fails, so the kernel compiles but deadlocks
#     at launch (`txStreamSynchronize` spins forever).
#   * `tl.trans` of a *computed* tensor (dot output / softmax result) and of a
#     regular strided load IS safe.
#   * A load whose *inner* (last) axis is the scattered index lowers to
#     per-element `rdma1d(count=1)` (~39 MB/s); a load whose inner axis is a
#     contiguous run of D elements lowers to one DMA per row (~3.9 GB/s).
#
# The standard layout `S = Q @ K^T` needs K^T = [D, BK], which forces the
# scattered index onto the inner axis -> the ~95x-slow path.  We therefore run
# attention *transposed* so that K==V is loaded only as [BK, D] (D contiguous on
# the inner axis, one DMA per token):
#
#   Q^T  = load(q_base + offs_d[:,None] + offs_h[None,:]*stride_qh)   [D, BH]
#   K    = load(kv_base + ids[:,None]*stride_kvn + offs_d[None,:])    [BK, D]
#   S^T  = K @ Q^T                                                     [BK, BH]
#   W    = softmax over axis 0 (tokens), online                        [BK, BH]
#   O    = W^T @ K                                                     [BH, D]
#
# The only transpose is `tl.trans(W)` of a computed softmax result (safe), and
# every load has D contiguous on the inner axis (fast).  Numerically identical
# to the generic `flag_gems.fused.flashmla_sparse` contract
# (output, max_logits, lse); attn_sink scales the output only.
#
# v1 scope: DQK=d_v=512 (no 64-dim RoPE tail), HQ in {64,128}, BK=64/BH=64.


@triton.jit
def _triton_flash_mla_sparse_fwd_tsingmicro(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qm,
    stride_qh,
    stride_kvn,
    stride_tm,
    stride_om,
    stride_oh,
    stride_mm,
    stride_lm,
    SQ,
    HQ: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    D: tl.constexpr = 512

    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)  # prevent mul overflow
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH

    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    max_log_base = max_logits + i_sq * stride_mm + gbh_base
    l_base = lse + i_sq * stride_lm + gbh_base

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, BK)

    # Q^T loaded directly in [D, BH] (d stride 1 on the row axis, h stride
    # stride_qh on the column axis) -- a regular strided load, no transpose.
    qt = tl.load(q_base + offs_d[:, None] + offs_h[None, :] * stride_qh)  # [D, BH]

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc = tl.zeros([BH, D], dtype=tl.float32)

    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)

    for ck in range(NK):
        # -- window ids, zeroing invalid slots (OOB-safe for the gather) --
        t_msk = ck * BK + offs_t < topk_len
        ids_raw = tl.load(t_base + ck * BK + offs_t, mask=t_msk, other=-1)
        valid = t_msk & (ids_raw >= 0) & (ids_raw < SKV)
        ids = tl.where(valid, ids_raw, 0).to(tl.int64)  # prevent mul overflow

        # -- K == V: one gather [BK, D] with D contiguous on the inner axis --
        K = tl.load(
            kv_base + ids[:, None] * stride_kvn + offs_d[None, :],
            cache_modifier=".cg",
        )  # [BK, D]

        # -- scores transposed [BK, BH] = K @ Q^T --
        st = tl.dot(K, qt, out_dtype=tl.float32)
        st *= sm_scale
        st = tl.where(valid[:, None], st, float("-inf"))

        # -- online softmax over axis 0 (tokens) --
        new_max = tl.maximum(max_log, tl.max(st, axis=0))  # [BH]
        exp_qtk = tl.exp(st - new_max[None, :])  # [BK, BH]
        sum_qk = tl.sum(exp_qtk, axis=0)  # [BH]
        alpha = tl.exp(max_log - new_max)  # [BH]
        sum_exp = sum_exp * alpha + sum_qk  # [BH]

        # -- O [BH, D] += W^T @ K; W^T is the only transpose (computed, safe) --
        Wt = tl.trans(exp_qtk)  # [BH, BK]
        acc = tl.dot(
            Wt.to(tl.bfloat16),
            K,
            acc * alpha[:, None],
            out_dtype=tl.float32,
        )

        max_log = new_max

    # -- epilogue --
    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    tl.store(max_log_base + offs_h, max_log)  # [BH] float32

    orig_lse = max_log + tl.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h, lse_out)  # [BH] float32

    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        sum_exp_new_lse = tl.exp(orig_lse) + tl.exp(sink)
        factor = tl.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp

    out_vals = tl.where(valid_mask[:, None], acc * factor[:, None], 0.0)
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]  # [BH, D]
    tl.store(o_ptr, out_vals.to(tl.bfloat16))


def _fmt_value(value):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, stride={tuple(value.stride())}, "
            f"dtype={value.dtype})"
        )
    return repr(value)


def log_args(func):
    """Print each argument on one line and the wall-clock execution time in ms.

    Gated behind ``FLAG_GEMS_LOG_FLASH_MLA_SPARSE=1`` so the per-call stdout
    formatting/flush doesn't tax the serving hot path.
    """

    enabled = os.getenv("FLAG_GEMS_LOG_FLASH_MLA_SPARSE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        prefix = f"[{func.__name__}]"
        for name, value in bound.arguments.items():
            print(f"{prefix} {name}: {_fmt_value(value)}", flush=True)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"{prefix} elapsed_ms={elapsed_ms:.3f}", flush=True)

    return wrapper


@log_args
def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """tsingmicro backend specialization of sparse MLA prefill.

    Drop-in replacement for the generic `flag_gems.fused.flashmla_sparse`; same
    signature and contract (output, max_logits, lse).  Runs attention transposed
    so K==V is gathered only as [BK, D] (D contiguous inner) and the sole
    transpose is of a computed softmax result; DQK=d_v=512 only.
    """
    logger.debug("GEMS_TSINGMICRO FLASH_MLA_SPARSE_FWD")
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
    )
    SQ, HQ, DQK = q.shape
    SKV = kv.shape[0]
    assert d_v == 512 and DQK == 512, (
        "tsingmicro flash_mla_sparse supports DQK=d_v=512 only; DQK=576 (RoPE "
        "tail) deferred — the latent/tail split breaks 4KB contiguity"
    )
    _, HKV, TOPK = indices.shape
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 64 or HQ == 128, "tsingmicro flash_mla_sparse assumes HQ in {64,128}"
    if attn_sink is not None:
        assert attn_sink.is_contiguous() and attn_sink.dtype == torch.float32
        assert attn_sink.shape == (HQ,)
    if topk_length is not None:
        assert topk_length.is_contiguous() and topk_length.dtype == torch.int32
        assert topk_length.shape == (SQ,)

    BK, BH = 64, 64
    output = torch.empty((SQ, HQ, d_v), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(HQ, BH) * SQ,)
    with torch_device_fn.device(device):
        _triton_flash_mla_sparse_fwd_tsingmicro[grid](
            q,
            kv,
            indices,
            attn_sink,
            topk_length,
            sm_scale,
            output,
            max_logits,
            lse,
            q.stride(0),
            q.stride(1),
            kv.stride(0),
            indices.stride(0),
            output.stride(0),
            output.stride(1),
            max_logits.stride(0),
            lse.stride(0),
            SQ,
            HQ,
            SKV,
            TOPK,
            attn_sink is not None,
            topk_length is not None,
            BK,
            BH,
            num_warps=8,
            num_stages=2,
        )

    return output, max_logits, lse
