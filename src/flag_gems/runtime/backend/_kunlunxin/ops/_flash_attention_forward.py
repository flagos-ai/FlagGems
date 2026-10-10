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
"""Kunlunxin fast path for aten::_flash_attention_forward.

The generic entry (flag_gems.ops._flash_attention_forward -> ops.attention ->
ops.flash_api.mha_fwd) rebuilds the full launch machinery on every call:
~50 fwd_params attribute writes, an args() tuple, several scratch allocations,
the libentry argument-classification loop and the triton autotuner layer.
On this backend that per-call cost is comparable to the remaining execution
time; for small shapes it dominates the operation.

On the first call for a given configuration this entry delegates through the
generic path and captures the deepest JITFunction.run arguments; later calls
replay them with the fresh input/output tensors spliced back in. Correctness
is bit-identical (same kernel, same arguments; verified diff=0).

Any doubt (unsupported config, capture mismatch, replay error) falls back to
the generic path; set FG_FWD_FASTPATH=0 to disable entirely.
"""

import importlib
import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("FG_FWD_FASTPATH", "1") != "0"

_LOCK = threading.Lock()
_CACHE = {}
_MAX_KEYS = 64
_GENERIC = None
_LAST = None


def _generic_entry():
    global _GENERIC
    if _GENERIC is None:
        gen = importlib.import_module("flag_gems.ops._flash_attention_forward")
        _GENERIC = gen._flash_attention_forward
    return _GENERIC


def _supported(query, key, value, cu_q, cu_k, dropout_p, seqused_k, alibi_slopes):
    return (
        cu_q is None
        and cu_k is None
        and seqused_k is None
        and alibi_slopes is None
        and (dropout_p or 0.0) == 0.0
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.is_cuda
    )


def _key_of(query, key, value, is_causal, scale, wl, wr, rdm):
    return (
        tuple(query.shape),
        tuple(query.stride()),
        query.dtype,
        tuple(key.shape),
        tuple(key.stride()),
        tuple(value.shape),
        tuple(value.stride()),
        bool(is_causal),
        None if scale is None else float(scale),
        wl,
        wr,
        bool(rdm),
    )


def _capture_and_delegate(entry, call_args, call_kwargs):
    fk = importlib.import_module("flag_gems.ops.flash_kernel")
    kern = getattr(fk, "flash_attn_fwd_kernel", None)
    if kern is None:
        kern = getattr(fk, "flash_fwd_kernel", None)
    jf = getattr(kern, "jit_function", None) if kern is not None else None
    cap = {}
    if jf is None:
        return entry(*call_args, **call_kwargs), cap
    orig = jf.run

    def hook(*a, **k):
        if "a" not in cap:
            cap["a"] = tuple(a)
            cap["kw"] = dict(k)
        return orig(*a, **k)

    jf.run = hook
    try:
        ret = entry(*call_args, **call_kwargs)
    finally:
        jf.run = orig
    cap["jf"] = jf
    return ret, cap


def _match_indices(a, tensors):
    """Map each tensor to the (unique) position in `a` sharing its data_ptr.

    Shape equality is NOT required: the generic chain may hand the kernel
    reshaped/transposed views (e.g. the GQA/decode qkv-group swap), which keep
    the caller's storage but change the logical shape. The downstream launch
    consumes the pointer (plus captured constexprs/strides), so replaying with
    the caller's own tensors is equivalent.
    """
    out = []
    for t in tensors:
        hit = None
        for i, x in enumerate(a):
            if isinstance(x, torch.Tensor) and x.data_ptr() == t.data_ptr():
                if hit is not None:
                    hit = None
                    break
                hit = i
        if hit is None:
            return None
        out.append(hit)
    return out


def _replay(rec, query, key, value, out, lse):
    a = list(rec["a"])
    a[rec["i_q"]] = query
    a[rec["i_k"]] = key
    a[rec["i_v"]] = value
    a[rec["i_o"]] = out
    a[rec["i_lse"]] = lse
    rec["jf"].run(*a, **rec["kw"])
    return (out, lse, rec["seed"], rec["offset"], rec["p"])


def _flash_attention_forward(
    query,
    key,
    value,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
    seqused_k=None,
    alibi_slopes=None,
):
    logger.debug("GEMS_KUNLUNXIN _FLASH_ATTENTION_FORWARD")
    entry = _generic_entry()

    def _delegate():
        return entry(
            query,
            key,
            value,
            cumulative_sequence_length_q,
            cumulative_sequence_length_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            return_debug_mask,
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            seqused_k=seqused_k,
            alibi_slopes=alibi_slopes,
        )

    if not _ENABLED or not _supported(
        query,
        key,
        value,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        dropout_p,
        seqused_k,
        alibi_slopes,
    ):
        return _delegate()

    global _LAST
    last = _LAST
    rec = None
    if last is not None and query is last[0] and key is last[1] and value is last[2]:
        fl = last[3]
        if (
            fl[0] == is_causal
            and fl[4] == return_debug_mask
            and fl[1] == scale
            and fl[2] == window_size_left
            and fl[3] == window_size_right
        ):
            rec = last[4]
    if rec is None:
        flags = (
            is_causal,
            scale,
            window_size_left,
            window_size_right,
            return_debug_mask,
        )
        kk = _key_of(
            query,
            key,
            value,
            is_causal,
            scale,
            window_size_left,
            window_size_right,
            return_debug_mask,
        )
        with _LOCK:
            rec = _CACHE.get(kk)
        if rec is not None:
            _LAST = (query, key, value, flags, rec)
    if rec is not None:
        try:
            out = torch.empty_like(query)
            lse = torch.empty_like(rec["lse_ref"])
            return _replay(rec, query, key, value, out, lse)
        except Exception:
            with _LOCK:
                _CACHE.pop(kk, None)

    ret, cap = _capture_and_delegate(
        entry,
        (
            query,
            key,
            value,
            cumulative_sequence_length_q,
            cumulative_sequence_length_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            return_debug_mask,
        ),
        dict(
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            seqused_k=seqused_k,
            alibi_slopes=alibi_slopes,
        ),
    )
    a = cap.get("a")
    if a and "jf" in cap and torch.is_tensor(ret[0]) and torch.is_tensor(ret[1]):
        idx = _match_indices(a, [query, key, value, ret[0], ret[1]])
        if idx is not None:
            rec = {
                "a": a,
                "kw": cap["kw"],
                "jf": cap["jf"],
                "i_q": idx[0],
                "i_k": idx[1],
                "i_v": idx[2],
                "i_o": idx[3],
                "i_lse": idx[4],
                "lse_ref": ret[1],
                "seed": ret[2],
                "offset": ret[3],
                "p": ret[4],
            }
            with _LOCK:
                if len(_CACHE) >= _MAX_KEYS:
                    _CACHE.clear()
                _CACHE[kk] = rec
            _LAST = (query, key, value, flags, rec)
    return ret
