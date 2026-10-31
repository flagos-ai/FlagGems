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

"""XPU (SDNN) override for ``triton_lighting_indexer_k_tiled_interface``.

The generic ``flag_gems/fused/DSA/indexer_k_tiled.py`` kernel does not lower on the
Triton-XPU backend: its ``for ck in range(CK, warp_specialize=True)`` dynamic trip
count deadlocks the SDNN pipeline (status 702) and its
``acc.trans().reshape(BK,BQ,H).sum(-1).trans()`` head reduction faults on the
single-column transpose (status 719).

This override expresses the same computation on the ``tle.dsa`` SDNN path:

* k tiles are staged through ``tle.pipe`` + ``tle.dsa.copy``, whose explicit
  ``pipe_commit/pipe_wait/pipe_release`` sync edges survive a dynamic trip count
  (instead of the auto pipeline's signal/wait derivation);
* the dot uses an explicit zeroed accumulator (SDNN "mma always accumulates") with
  ``out_dtype=f16``;
* ``tl.maximum(acc, tl.zeros_like(acc))`` keeps the relu in f16 (a bare ``0.0``
  literal would promote to f32 and re-trigger the broadcast bug);
* the head reduction is ``tl.trans(acc).sum(-1)`` -> ``[BK]`` 1D, avoiding the
  ``[BK,1] -> [1,BK]`` single-column transpose.

bf16 input is converted to f16 on the host before the kernel: a bf16 ``tle.dsa``
buffer does not build, and the in-kernel bf16->f16 dma-in is not expressible via
``tle.dsa.copy``.  This matches the generic kernel's own ``.to(tl.float16)`` casts.
"""

import sys

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

_BQ = 1
_BK = 64
_TK = 2048
_CAPACITY = 4


@triton.jit
def _lighting_indexer_k_tiled_tle(
    q_index,
    k_index,
    cu_bg_seqlens,
    cu_ed_seqlens,
    weights,
    logits,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_wh,
    stride_wbk,
    stride_lm,
    stride_ln,
    Q: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    TK: tl.constexpr,
    D: tl.constexpr,
    CU: tl.constexpr,
    BK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    i_sh = tl.program_id(0)
    i_k = tl.program_id(1)

    bos_val = tl.load(cu_bg_seqlens + i_sh, mask=i_sh < CU, other=1000000000) + i_k * TK
    eos_val = tl.load(cu_ed_seqlens + i_sh, mask=i_sh < CU, other=-1000000000)
    eos_val = tl.minimum(eos_val, bos_val + (i_k + 1) * TK)
    bos = tl.maximum(bos_val, 0)
    eos = tl.minimum(eos_val, K)
    CK = eos - bos

    if CK > 0:
        offs_h = tl.arange(0, H) + i_sh * H
        offs_d = tl.arange(0, D)

        q_ptr = q_index + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q_buf = tle.dsa.alloc([H, D], tl.float16, tle.dsa.UNI_SRAM)
        tle.dsa.copy(q_ptr, q_buf)
        q_blk = tle.dsa.to_tensor(q_buf)

        # w is pre-broadcast to [Q, H, BK] on the host so the multiply below is
        # element-wise; the [H,1] broadcast-mul mis-lowers when the other operand
        # is the relu'd MMA accumulator (M=64).
        offs_bk0 = tl.arange(0, BK)
        w_ptr = weights + offs_h[:, None] * stride_wh + offs_bk0[None, :] * stride_wbk
        w_buf = tle.dsa.alloc([H, BK], tl.float16, tle.dsa.UNI_SRAM)
        tle.dsa.copy(w_ptr, w_buf)
        w_blk = tle.dsa.to_tensor(w_buf)

        k_buf = tle.dsa.alloc([CAPACITY, BK, D], tl.float16, tle.dsa.UNI_SRAM)
        k_pipe = tle.pipe(capacity=CAPACITY, scope="core", name="k", tile=k_buf)
        k_writer = k_pipe.writer()
        k_reader = k_pipe.reader()

        CK_blocks = tl.cdiv(CK, BK)
        for ck in range(CK_blocks):
            offs_bk = ck * BK + tl.arange(0, BK)
            k_pos = bos + offs_bk
            k_tail = tl.minimum(eos - (bos + ck * BK), BK)
            k_ptr = k_index + k_pos[:, None] * stride_kn + offs_d[None, :] * stride_kd

            k_slot = k_writer.acquire(ck)
            tle.dsa.copy(k_ptr, k_slot.tile, sizes=[k_tail, D])
            k_writer.commit(ck)

            k_ready = k_reader.wait(ck)
            k_t = tl.trans(tle.dsa.to_tensor(k_ready.slot.tile))  # [D, BK]

            acc_buf = tle.dsa.alloc([H, BK], tl.float16, tle.dsa.UNI_SRAM)
            tle.dsa.fill(acc_buf, 0.0)
            acc = tl.dot(q_blk, k_t, acc=tle.dsa.to_tensor(acc_buf), out_dtype=tl.float16)
            acc = tl.maximum(acc, tl.zeros_like(acc)) * w_blk
            out = tl.trans(acc).sum(-1)  # [BK]
            k_reader.release(ck)

            out_ptr = logits + i_sh * stride_lm + k_pos * stride_ln
            tl.store(out_ptr, out.to(tl.float32), mask=k_pos < eos)


def triton_lighting_indexer_k_tiled_interface_xpu(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke
):
    Q, H, D = q.shape[0], q.shape[1], q.shape[2]
    K = kv.shape[0]
    CU = cu_seqlen_ks.shape[0]
    logits = torch.full([Q, K], float("-inf"), device="cuda", dtype=torch.float32)

    # bf16 -> f16 / f32 -> f16 on the host: bf16 tle.dsa buffers do not build, and
    # the in-kernel bf16->f16 dma-in is not expressible via tle.dsa.copy.  This is
    # the same f16 cast the generic kernel does via `.to(tl.float16)`.
    q_f16 = q.to(torch.float16)
    kv_f16 = kv.to(torch.float16)
    # weights [Q, H] -> [Q, H, BK] pre-broadcast: the kernel multiplies
    # element-wise (see the kernel's w load); this also lands f16.
    w_bcast = (
        weights.unsqueeze(-1).expand(Q, H, _BK).contiguous().to(torch.float16)
    )

    NQ = triton.cdiv(Q, _BQ)
    NK = triton.cdiv(K, _TK)
    grid = (NQ, NK)
    _lighting_indexer_k_tiled_tle[grid](
        q_f16,
        kv_f16,
        cu_seqlen_ks,
        cu_seqlen_ke,
        w_bcast,
        logits,
        q_f16.stride(1),
        q_f16.stride(2),
        kv_f16.stride(0),
        kv_f16.stride(1),
        w_bcast.stride(1),
        w_bcast.stride(2),
        logits.stride(0),
        logits.stride(1),
        Q,
        H,
        K,
        _TK,
        D,
        CU,
        _BK,
        _CAPACITY,
        is_sdnn=True,
        num_stages=2,
    )
    return logits


def _install():
    """Replace ``flag_gems.fused.DSA.indexer_k_tiled.triton_lighting_indexer_k_tiled_interface``
    with the XPU implementation.  The DSA indexer is called via direct module import in
    tests/test_DSA/test_indexer_k_tiled.py, so the SpecOpRegistrar namespace swap cannot
    reach it.  Unlike bin_topk/sparse_mla, ``flag_gems.fused.DSA.indexer_k_tiled`` is not
    imported by ``flag_gems/fused/__init__.py``, so it is force-imported here before
    patching its attribute."""
    import importlib

    gmod = sys.modules.get("flag_gems.fused.DSA.indexer_k_tiled")
    if gmod is None:
        gmod = importlib.import_module("flag_gems.fused.DSA.indexer_k_tiled")
    gmod.triton_lighting_indexer_k_tiled_interface = (
        triton_lighting_indexer_k_tiled_interface_xpu
    )


_install()
