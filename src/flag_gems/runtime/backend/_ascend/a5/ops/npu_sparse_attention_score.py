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

import struct

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


def _bf16_rne_host(x):
    """Round a Python float to the nearest BF16 value (round-to-nearest-even),
    expressed as a float32 so it can be passed to the kernel as a scalar."""
    bits = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    if (bits & 0x7FFFFFFF) > 0x7F800000:  # NaN / Inf: leave untouched
        return float(x)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@triton.jit
def _rne_bf16(x):
    """Explicit round-to-nearest-even FP32 -> BF16 as a bit transform, returned
    widened to FP32. Written this way because the compiler is free to fold an
    observable bf16 intermediate (``v.to(tl.bfloat16).to(tl.float32)``) out of a
    pure-FP32 expression, which would silently drop the rounding the native
    kernel performs."""
    b = x.to(tl.int32, bitcast=True)
    b = (b + 32767 + ((b >> 16) & 1)) & -65536
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _pos_info(
    t,
    asl_ptr,
    aslkv_ptr,
    ASLS: tl.constexpr,
    ASLKS: tl.constexpr,
    NASL: tl.constexpr,
    HAS_ASL: tl.constexpr,
    HAS_ASLKV: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
):
    """Sequence index of token ``t``, plus the inclusive-causal boundary of its
    diagonal KV block: the logical id ``last`` of that block and the last key
    lane ``limit`` that participates. ``limit == BS - 1`` means the diagonal
    block is complete, so the mask is a no-op for that token."""
    if HAS_ASL:
        s = 0
        cum = 0
        start = 0
        for i in tl.static_range(NASL):
            cum = cum + tl.load(asl_ptr + i * ASLS).to(tl.int32)
            ge = (t >= cum).to(tl.int32)
            start = tl.where(ge != 0, cum, start)
            s = s + ge
        s = tl.minimum(s, S - 1)
    else:
        s = 0
        start = 0
    # token position inside its own sequence
    qi = t - start
    if HAS_ASLKV:
        kvlen = tl.load(aslkv_ptr + s * ASLKS).to(tl.int32)
        qlen = tl.load(asl_ptr + s * ASLS).to(tl.int32)
        hist = tl.maximum(kvlen - qlen, 0)
    else:
        hist = 0
    diag = hist + qi
    last = diag // BS
    limit = diag - last * BS
    return s, last, limit


@triton.jit
def _resolve(
    sel_base, bt_base, b, BTW: tl.constexpr, NB: tl.constexpr, SELB: tl.constexpr
):
    """Logical then physical id of the b-th selected block of this token.

    Two dependent scalar loads; resolving one block ahead lets the compiler issue
    them while the current block's K/V transfers and dots are in flight, instead
    of gating each block's tile address on a fresh L2 round trip."""
    idx = tl.load(sel_base + b * SELB).to(tl.int32)
    idx = tl.maximum(tl.minimum(idx, BTW - 1), 0)
    page = tl.load(bt_base + idx).to(tl.int32)
    page = tl.maximum(tl.minimum(page, NB - 1), 0)
    return idx, page


@triton.jit
def _sas_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    sel_ptr,
    bt_ptr,
    snum_ptr,
    asl_ptr,
    aslkv_ptr,
    out_ptr,
    scale_bf,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    TOPK: tl.constexpr,
    BTW: tl.constexpr,
    NB: tl.constexpr,
    S: tl.constexpr,
    NASL: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KD0: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VD0: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OD0: tl.constexpr,
    SELT: tl.constexpr,
    SELB: tl.constexpr,
    SNT: tl.constexpr,
    ASLS: tl.constexpr,
    ASLKS: tl.constexpr,
    HAS_SNUM: tl.constexpr,
    HAS_ASL: tl.constexpr,
    HAS_ASLKV: tl.constexpr,
):
    t = tl.program_id(0)

    s, last, limit = _pos_info(
        t, asl_ptr, aslkv_ptr, ASLS, ASLKS, NASL, HAS_ASL, HAS_ASLKV, S, BS
    )

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, BS)
    k_off = offs_s[:, None] * KS1 + offs_d[None, :] * KD0
    v_off = offs_s[:, None] * VS1 + offs_d[None, :] * VD0
    # inclusive-causal lane mask of the diagonal block: key j participates iff
    # j <= limit (blocks before the diagonal keep every lane)
    keep = offs_s[None, :] <= limit

    # ---- query tile for all heads of this token: [H, D]
    q = tl.load(q_ptr + t * QS0 + offs_h[:, None] * QS1 + offs_d[None, :] * QS2)

    if HAS_SNUM:
        n = tl.load(snum_ptr + t * SNT).to(tl.int32)
    else:
        n = TOPK
    n = tl.minimum(tl.maximum(n, 0), TOPK)

    sel_base = sel_ptr + t * SELT
    bt_base = bt_ptr + s * BTW

    m_i = tl.full([H], -float("inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D], tl.float32)

    # resolve the first selected block before the loop: its tile address is then
    # ready when the loop body starts, and every later block's ids are resolved
    # one iteration ahead (see _resolve)
    idx_c, page_c = _resolve(sel_base, bt_base, 0, BTW, NB, SELB)

    # One page per online-softmax update, in the original selected order: the
    # running maximum and the accumulator rescales it implies must follow the
    # native page sequence, because the BF16 rounding of the probabilities
    # depends on the shift they are computed against.
    for b in range(0, n):
        idx = idx_c
        page = page_c
        idx_c, page_c = _resolve(
            sel_base, bt_base, tl.minimum(b + 1, n - 1), BTW, NB, SELB
        )

        kt = tl.load(k_ptr + page * KS0 + k_off)
        vt = tl.load(v_ptr + page * VS0 + v_off)

        # The first round uses the native convert instead of the five-op integer
        # RNE transform: a focused target A/B (tmp/debug/probe_score_variants.py)
        # measured this substitution at 0 differing output bits over 196608
        # elements, while the same probe showed a bf16 multiply does NOT round
        # like RNE (52% of elements differ) and that the store convert is also
        # bit-equal. The second round therefore keeps the explicit transform.
        sc = _rne_bf16(
            tl.dot(q, tl.trans(kt)).to(tl.bfloat16).to(tl.float32) * scale_bf
        )
        # invalid lanes never take part in the maximum nor in the sums
        sc = tl.where(keep | (idx != last), sc, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(sc, 1))
        # degenerate guard: only reachable for an all-masked page, which cannot
        # happen while the diagonal block keeps lane 0; inert otherwise
        m_new = tl.where(m_new > -float("inf"), m_new, 0.0)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        # denominator sums the PRE-cast FP32 probabilities
        l_i = l_i * alpha + tl.sum(p, 1)
        # the P@V dot consumes the BF16-rounded probabilities, FP32 accumulation
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    if out_ptr.dtype.element_ty == tl.bfloat16:
        out = _rne_bf16(out)
    tl.store(
        out_ptr + t * OS0 + offs_h[:, None] * OS1 + offs_d[None, :] * OD0,
        out.to(out_ptr.dtype.element_ty),
    )


def _pow2(x, minimum=1):
    v = minimum
    while v < x:
        v *= 2
    return v


def _launch(
    query,
    key,
    value,
    select_idx,
    block_table,
    *,
    select_num_idx=None,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    q_input_layout="TND",
    kv_input_layout="BNSD",
    num_key_value_heads=1,
    scale_value=1.0,
    block_size=128,
    top_k=16,
    inner_precise=0,
    attention_out_dtype=None,
):
    T, H, D = query.shape
    # KV tensors are (num_blocks, block_len, num_kv_heads==1, D) in the frozen
    # ABI; derive the element strides of one gathered block from the tensor.
    if key.dim() == 4:
        BS = _pow2(int(key.shape[1]))
        KS1 = key.stride(1)
        VS1 = value.stride(1)
    else:
        BS = _pow2(int(block_size))
        KS1 = key.stride(-2)
        VS1 = value.stride(-2)
    KD0 = key.stride(-1)
    VD0 = value.stride(-1)
    TOPK = int(select_idx.shape[-1])
    BTW = int(block_table.shape[-1])
    S = int(block_table.shape[0])
    NB = int(key.shape[0])
    out_dtype = query.dtype if attention_out_dtype is None else attention_out_dtype
    out = torch.empty((T, H, D), dtype=out_dtype, device=query.device)
    if T == 0:
        return out

    Hp = _pow2(H)
    Dp = _pow2(D)
    NASL = 0 if actual_seq_lengths is None else int(actual_seq_lengths.numel())
    scale_bf = _bf16_rne_host(scale_value)

    _sas_kernel[(T,)](
        query,
        key,
        value,
        select_idx,
        block_table,
        select_num_idx if select_num_idx is not None else query,
        actual_seq_lengths if actual_seq_lengths is not None else query,
        actual_seq_lengths_kv if actual_seq_lengths_kv is not None else query,
        out,
        scale_bf,
        H=Hp,
        D=Dp,
        BS=BS,
        TOPK=TOPK,
        BTW=BTW,
        NB=NB,
        S=S,
        NASL=NASL,
        QS0=query.stride(0),
        QS1=query.stride(1),
        QS2=query.stride(2),
        KS0=key.stride(0),
        KS1=KS1,
        KD0=KD0,
        VS0=value.stride(0),
        VS1=VS1,
        VD0=VD0,
        OS0=out.stride(0),
        OS1=out.stride(1),
        OD0=out.stride(2),
        SELT=select_idx.stride(-2),
        SELB=select_idx.stride(-1),
        SNT=1 if select_num_idx is None else select_num_idx.stride(-1),
        ASLS=1 if actual_seq_lengths is None else actual_seq_lengths.stride(0),
        ASLKS=1 if actual_seq_lengths_kv is None else actual_seq_lengths_kv.stride(0),
        HAS_SNUM=select_num_idx is not None,
        HAS_ASL=actual_seq_lengths is not None,
        HAS_ASLKV=(
            actual_seq_lengths_kv is not None and actual_seq_lengths is not None
        ),
    )
    return out


def npu_sparse_attention_score(
    query,
    key,
    value,
    select_idx,
    block_table,
    *,
    select_num_idx=None,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    q_input_layout="TND",
    kv_input_layout="BNSD",
    num_key_value_heads=1,
    scale_value=1.0,
    block_size=128,
    top_k=16,
    inner_precise=0,
    attention_out_dtype=None,
):
    """Inclusive causal sparse BF16 attention over paged key and value tensors."""
    if (
        query.ndim != 3
        or query.shape[1:] != (16, 128)
        or key.ndim != 4
        or key.shape[1:] != (128, 1, 128)
        or value.shape != key.shape
        or block_size != 128
        or num_key_value_heads != 1
        or q_input_layout != "TND"
        or kv_input_layout != "BNSD"
        or inner_precise != 4
        or top_k != 16
        or attention_out_dtype not in (None, query.dtype)
        or any(
            x is not None for x in (q_dequant_scale, k_dequant_scale, v_dequant_scale)
        )
    ):
        raise ValueError(
            "expected TND BF16 sparse attention, 16 heads, 128 lanes and one KV head"
        )
    if (
        select_idx.shape != (1, query.shape[0], 16)
        or block_table.ndim != 2
        or select_num_idx is None
        or select_num_idx.shape != (1, query.shape[0])
        or actual_seq_lengths is None
        or actual_seq_lengths_kv is None
        or actual_seq_lengths.shape != (block_table.shape[0],)
        or actual_seq_lengths_kv.shape != actual_seq_lengths.shape
    ):
        raise ValueError("invalid page table, lengths, or sparse selection shape")
    if (
        any(t.dtype != torch.bfloat16 for t in (query, key, value))
        or any(
            t.dtype != torch.int32 or t.device != query.device or not t.is_contiguous()
            for t in (
                select_idx,
                block_table,
                select_num_idx,
                actual_seq_lengths,
                actual_seq_lengths_kv,
            )
        )
        or not all(t.is_contiguous() for t in (query, key, value))
    ):
        raise ValueError("expected contiguous BF16 data and contiguous int32 indices")
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
            select_idx,
            block_table,
            select_num_idx=select_num_idx,
            q_dequant_scale=q_dequant_scale,
            k_dequant_scale=k_dequant_scale,
            v_dequant_scale=v_dequant_scale,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            q_input_layout=q_input_layout,
            kv_input_layout=kv_input_layout,
            num_key_value_heads=num_key_value_heads,
            scale_value=scale_value,
            block_size=block_size,
            top_k=top_k,
            inner_precise=inner_precise,
            attention_out_dtype=attention_out_dtype,
        )
