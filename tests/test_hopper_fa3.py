"""
Benchmark + correctness for the migrated FlagGems FA3 path (Hopper backend)
against vLLM's vendored FA3.

Backends compared:
  - FlagGems            (flag_gems.flash_attn_varlen_func; uses the default
                         dispatch -- on Hopper this routes to the migrated
                         FA3 override, on other archs it falls back to v2)
  - vLLM-FA             (vllm.vllm_flash_attn, can dispatch FA3)  <- REFERENCE
  - torch varlen_attn   (PT 2.10+; non-paged only)

Correctness reference: vLLM's flash_attn_varlen_func output (FA3 by default,
fa_version=2 if --vllm-fa-version 2). vLLM-FA is the production reference
implementation we want our FlagGems FA3 to match bit-equivalently within
fp16/bf16 tolerance.

If vLLM is unavailable the script falls back to eager fp32 per-sequence SDPA
so it still runs; set --strict to force it to error out instead.

Run:
    python tools/bench_flag_gems_fa3_vs_vllm.py
    python tools/bench_flag_gems_fa3_vs_vllm.py --shapes prefill
    python tools/bench_flag_gems_fa3_vs_vllm.py --shapes paged
    python tools/bench_flag_gems_fa3_vs_vllm.py --vllm-fa-version 2
    python tools/bench_flag_gems_fa3_vs_vllm.py --no-correctness
"""

import argparse
import inspect
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import triton

import flag_gems

# =====================================================================
# is_fa3_supported -- try the migrated location first, then legacy
# =====================================================================
try:
    from flag_gems.runtime.backend._nvidia.hopper.ops.flash_api_v3 import (
        is_fa3_supported,
    )

    _FA3_SUPPORT_FROM = "hopper backend"
except ImportError:
    try:
        from flag_gems.ops.flash_api_v3 import is_fa3_supported  # legacy

        _FA3_SUPPORT_FROM = "legacy ops/"
    except ImportError:

        def is_fa3_supported() -> bool:
            if not torch.cuda.is_available():
                return False
            return torch.cuda.get_device_capability()[0] >= 9

        _FA3_SUPPORT_FROM = "fallback (cap check only)"


# =====================================================================
# Optional external backends
# =====================================================================
try:
    from torch.nn.attention.varlen import varlen_attn as torch_varlen_attn

    HAS_TORCH_VARLEN = True
    _torch_varlen_params = set(inspect.signature(torch_varlen_attn).parameters.keys())
    TORCH_VARLEN_HAS_WINDOW = "window_size" in _torch_varlen_params
    TORCH_VARLEN_HAS_IS_CAUSAL = "is_causal" in _torch_varlen_params
    TORCH_VARLEN_SCALE_KWARG: Optional[str] = None
    for cand in ("scale", "softmax_scale", "scale_factor"):
        if cand in _torch_varlen_params:
            TORCH_VARLEN_SCALE_KWARG = cand
            break
except ImportError:
    HAS_TORCH_VARLEN = False
    TORCH_VARLEN_HAS_WINDOW = False
    TORCH_VARLEN_HAS_IS_CAUSAL = False
    TORCH_VARLEN_SCALE_KWARG = None

try:
    from vllm.vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as vllm_fa_varlen,
    )

    HAS_VLLM_FA = True
    _vllm_fa_params = set(inspect.signature(vllm_fa_varlen).parameters.keys())
    VLLM_FA_HAS_FA_VERSION = "fa_version" in _vllm_fa_params
    VLLM_FA_HAS_BLOCK_TABLE = "block_table" in _vllm_fa_params
    VLLM_FA_HAS_SEQUSED_K = "seqused_k" in _vllm_fa_params
except ImportError:
    HAS_VLLM_FA = False
    VLLM_FA_HAS_FA_VERSION = False
    VLLM_FA_HAS_BLOCK_TABLE = False
    VLLM_FA_HAS_SEQUSED_K = False


# =====================================================================
# Shape presets
# =====================================================================
@dataclass
class Shape:
    name: str
    seq_lens: List[Tuple[int, int]]
    nh_q: int
    nh_k: int
    head_dim: int
    causal: bool
    paged: bool = False
    block_size: int = 16
    overcommit: float = 1.5


def prefill_shapes() -> List[Shape]:
    return [
        Shape("prefill_b4_s2k_d128_mha", [(2048, 2048)] * 4, 32, 32, 128, True),
        Shape("prefill_b4_s4k_d128_mha", [(4096, 4096)] * 4, 32, 32, 128, True),
        Shape("prefill_b4_s8k_d128_mha", [(8192, 8192)] * 4, 32, 32, 128, True),
        Shape("prefill_b2_s16k_d128_mha", [(16384, 16384)] * 2, 32, 32, 128, True),
        Shape("prefill_b4_s4k_d128_gqa4", [(4096, 4096)] * 4, 32, 8, 128, True),
        Shape("prefill_b4_s8k_d128_gqa4", [(8192, 8192)] * 4, 32, 8, 128, True),
        Shape("prefill_b8_s2k_d64_mha", [(2048, 2048)] * 8, 16, 16, 64, False),
    ]


def decode_shapes() -> List[Shape]:
    return [
        Shape("decode_b16_kv1k_d128_gqa4", [(1, 1024)] * 16, 32, 8, 128, True),
        Shape(
            "decode_b16_mixed_d128_gqa4",
            [(1, 512), (1, 1024), (1, 2048), (1, 4096)] * 4,
            32,
            8,
            128,
            True,
        ),
        Shape("decode_b32_kv2k_d128_gqa4", [(1, 2048)] * 32, 32, 8, 128, True),
    ]


def varlen_mixed_shapes() -> List[Shape]:
    return [
        Shape(
            "varlen_mixed_d128_gqa4",
            [(2048, 2048), (1, 4096), (1, 4096), (1024, 1024), (1, 8192), (1, 1024)],
            32,
            8,
            128,
            True,
        ),
        Shape(
            "varlen_serve_b32_1pf_31dec_d128_gqa4",
            [(2048, 2048)] + [(1, 1024 + 64 * i) for i in range(31)],
            32,
            8,
            128,
            True,
        ),
        Shape(
            "varlen_longtail_d128_gqa4",
            [(16384, 16384)] + [(256, 256)] * 16,
            32,
            8,
            128,
            True,
        ),
    ]


def paged_shapes() -> List[Shape]:
    return [
        Shape(
            "paged_decode_b16_kvmix_bs16_d128_gqa4",
            [(1, 1024 + 256 * i) for i in range(16)],
            32,
            8,
            128,
            True,
            paged=True,
            block_size=16,
        ),
        Shape(
            "paged_decode_b64_bs16_d128_gqa4",
            [(1, 512 + 128 * i) for i in range(64)],
            32,
            8,
            128,
            True,
            paged=True,
            block_size=16,
        ),
        Shape(
            "paged_serve_b32_1pf_31dec_bs16_d128_gqa4",
            [(2048, 2048)] + [(1, 1024 + 96 * i) for i in range(31)],
            32,
            8,
            128,
            True,
            paged=True,
            block_size=16,
        ),
        Shape(
            "paged_uniform_b4_s4k_bs16_d128_mha",
            [(4096, 4096)] * 4,
            32,
            32,
            128,
            True,
            paged=True,
            block_size=16,
        ),
    ]


def all_shapes() -> List[Shape]:
    return prefill_shapes() + decode_shapes() + varlen_mixed_shapes() + paged_shapes()


# =====================================================================
# Tensor builders
# =====================================================================
@dataclass
class Tensors:
    q: torch.Tensor
    k: torch.Tensor  # dense [total_k, Hk, D] or paged cache
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: Optional[torch.Tensor]
    seqused_k: torch.Tensor
    block_table: Optional[torch.Tensor]
    max_seqlen_q: int
    max_seqlen_k: int


def make_varlen(shape: Shape, dtype, device, seed: int = 0) -> Tensors:
    if shape.paged:
        return _make_paged_varlen(shape, dtype, device, seed)
    return _make_dense_varlen(shape, dtype, device, seed)


def _make_dense_varlen(shape, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    cu_q = [0]
    cu_k = [0]
    for q_len, k_len in shape.seq_lens:
        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + k_len)
    total_q = cu_q[-1]
    total_k = cu_k[-1]
    q = (
        torch.randn(
            (total_q, shape.nh_q, shape.head_dim),
            dtype=dtype,
            device=device,
            generator=g,
        )
        * 0.5
    )
    k = (
        torch.randn(
            (total_k, shape.nh_k, shape.head_dim),
            dtype=dtype,
            device=device,
            generator=g,
        )
        * 0.5
    )
    v = (
        torch.randn(
            (total_k, shape.nh_k, shape.head_dim),
            dtype=dtype,
            device=device,
            generator=g,
        )
        * 0.5
    )
    return Tensors(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
        seqused_k=torch.tensor(
            [s[1] for s in shape.seq_lens], dtype=torch.int32, device=device
        ),
        block_table=None,
        max_seqlen_q=max(s[0] for s in shape.seq_lens),
        max_seqlen_k=max(s[1] for s in shape.seq_lens),
    )


def _make_paged_varlen(shape, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    cpu_g = torch.Generator().manual_seed(seed + 1)
    bs = shape.block_size

    cu_q = [0]
    for q_len, _ in shape.seq_lens:
        cu_q.append(cu_q[-1] + q_len)
    total_q = cu_q[-1]
    q = (
        torch.randn(
            (total_q, shape.nh_q, shape.head_dim),
            dtype=dtype,
            device=device,
            generator=g,
        )
        * 0.5
    )

    blocks_per_req = [(k_len + bs - 1) // bs for _, k_len in shape.seq_lens]
    max_blocks_per_req = max(blocks_per_req)
    total_virtual_blocks = sum(blocks_per_req)
    num_physical_blocks = max(1, int(total_virtual_blocks * shape.overcommit))

    perm = torch.randperm(num_physical_blocks, generator=cpu_g)[:total_virtual_blocks]
    block_table = torch.zeros(
        (len(shape.seq_lens), max_blocks_per_req), dtype=torch.int32, device=device
    )
    cur = 0
    for r, n in enumerate(blocks_per_req):
        block_table[r, :n] = perm[cur : cur + n].to(torch.int32).to(device)
        cur += n

    k_cache = (
        torch.randn(
            (num_physical_blocks, bs, shape.nh_k, shape.head_dim),
            dtype=dtype,
            device=device,
            generator=g,
        )
        * 0.5
    )
    v_cache = torch.randn_like(k_cache)

    seqused_k = torch.tensor(
        [k_len for _, k_len in shape.seq_lens], dtype=torch.int32, device=device
    )

    return Tensors(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=block_table,
        max_seqlen_q=max(s[0] for s in shape.seq_lens),
        max_seqlen_k=max(s[1] for s in shape.seq_lens),
    )


def _gather_paged_to_dense(k_cache, v_cache, block_table, seqused_k):
    """Gather a paged K/V cache into per-sequence dense tensors.

    Returns lists of [k_len, Hk, D] tensors (one per request).
    """
    bs = k_cache.size(1)
    seqs_k = []
    seqs_v = []
    for r in range(seqused_k.numel()):
        k_len = int(seqused_k[r].item())
        if k_len == 0:
            seqs_k.append(k_cache.new_zeros((0, k_cache.size(2), k_cache.size(3))))
            seqs_v.append(v_cache.new_zeros((0, v_cache.size(2), v_cache.size(3))))
            continue
        n_blocks = (k_len + bs - 1) // bs
        block_ids = block_table[r, :n_blocks].to(torch.long)
        # [n_blocks, bs, Hk, D] -> [n_blocks*bs, Hk, D]
        k_flat = k_cache.index_select(0, block_ids).reshape(
            -1, k_cache.size(2), k_cache.size(3)
        )
        v_flat = v_cache.index_select(0, block_ids).reshape(
            -1, v_cache.size(2), v_cache.size(3)
        )
        seqs_k.append(k_flat[:k_len])
        seqs_v.append(v_flat[:k_len])
    return seqs_k, seqs_v


# =====================================================================
# Per-backend runners
# =====================================================================
def run_flag_gems(t: Tensors, shape: Shape):
    """Natural dispatch -- no fa_version passed. On Hopper the override
    picks FA3; on other archs the generic wrapper picks FA2."""
    sm_scale = 1.0 / math.sqrt(shape.head_dim)
    kwargs = dict(
        max_seqlen_q=t.max_seqlen_q,
        cu_seqlens_q=t.cu_seqlens_q,
        max_seqlen_k=t.max_seqlen_k,
        softmax_scale=sm_scale,
        causal=shape.causal,
    )
    if shape.paged:
        kwargs["seqused_k"] = t.seqused_k
        kwargs["block_table"] = t.block_table
    else:
        kwargs["cu_seqlens_k"] = t.cu_seqlens_k
    return flag_gems.flash_attn_varlen_func(t.q, t.k, t.v, **kwargs)


def run_torch_varlen(t: Tensors, shape: Shape):
    if shape.paged:
        raise NotImplementedError("torch_varlen_attn doesn't support paged KV")
    sm_scale = 1.0 / math.sqrt(shape.head_dim)
    if shape.nh_q != shape.nh_k:
        assert shape.nh_q % shape.nh_k == 0
        groups = shape.nh_q // shape.nh_k
        k_expanded = t.k.repeat_interleave(groups, dim=1)
        v_expanded = t.v.repeat_interleave(groups, dim=1)
    else:
        k_expanded, v_expanded = t.k, t.v

    extra = {}
    if TORCH_VARLEN_SCALE_KWARG is not None:
        extra[TORCH_VARLEN_SCALE_KWARG] = sm_scale
    if TORCH_VARLEN_HAS_WINDOW:
        extra["window_size"] = (-1, 0) if shape.causal else (-1, -1)
    elif TORCH_VARLEN_HAS_IS_CAUSAL:
        extra["is_causal"] = shape.causal
    elif shape.causal:
        raise RuntimeError(
            "this PyTorch build's varlen_attn doesn't expose a causal flag"
        )
    return torch_varlen_attn(
        t.q,
        k_expanded,
        v_expanded,
        t.cu_seqlens_q,
        t.cu_seqlens_k,
        t.max_seqlen_q,
        t.max_seqlen_k,
        **extra,
    )


def run_vllm_fa(t: Tensors, shape: Shape, fa_version: int = 3):
    sm_scale = 1.0 / math.sqrt(shape.head_dim)
    extra = dict(softmax_scale=sm_scale, causal=shape.causal)
    if VLLM_FA_HAS_FA_VERSION:
        extra["fa_version"] = fa_version

    if shape.paged:
        if not VLLM_FA_HAS_BLOCK_TABLE:
            raise NotImplementedError("vllm flash_attn doesn't support block_table")
        if not VLLM_FA_HAS_SEQUSED_K:
            raise NotImplementedError("vllm flash_attn build lacks seqused_k kwarg")
        return vllm_fa_varlen(
            t.q,
            t.k,
            t.v,
            max_seqlen_q=t.max_seqlen_q,
            cu_seqlens_q=t.cu_seqlens_q,
            max_seqlen_k=t.max_seqlen_k,
            seqused_k=t.seqused_k,
            block_table=t.block_table,
            **extra,
        )
    return vllm_fa_varlen(
        t.q,
        t.k,
        t.v,
        max_seqlen_q=t.max_seqlen_q,
        cu_seqlens_q=t.cu_seqlens_q,
        max_seqlen_k=t.max_seqlen_k,
        cu_seqlens_k=t.cu_seqlens_k,
        **extra,
    )


# =====================================================================
# Eager reference (fp32, per-sequence). Slow but unambiguously correct.
# Handles dense and paged uniformly via _gather_paged_to_dense.
# =====================================================================
def eager_reference(t: Tensors, shape: Shape) -> torch.Tensor:
    sm_scale = 1.0 / math.sqrt(shape.head_dim)
    cu_q = t.cu_seqlens_q
    if shape.paged:
        seqs_k, seqs_v = _gather_paged_to_dense(t.k, t.v, t.block_table, t.seqused_k)
    else:
        seqs_k = []
        seqs_v = []
        cu_k = t.cu_seqlens_k
        for b in range(cu_q.numel() - 1):
            seqs_k.append(t.k[cu_k[b] : cu_k[b + 1]])
            seqs_v.append(t.v[cu_k[b] : cu_k[b + 1]])

    outs = []
    for b in range(cu_q.numel() - 1):
        q_b = t.q[cu_q[b] : cu_q[b + 1]]  # [Sq, Hq, D]
        k_b = seqs_k[b]  # [Sk, Hk, D]
        v_b = seqs_v[b]
        Sq, Hq, D = q_b.shape
        Sk = k_b.size(0)
        if k_b.size(1) != Hq:
            assert Hq % k_b.size(1) == 0
            rep = Hq // k_b.size(1)
            k_b = k_b.repeat_interleave(rep, dim=1)
            v_b = v_b.repeat_interleave(rep, dim=1)
        qh = q_b.transpose(0, 1).float()  # [H, Sq, D]
        kh = k_b.transpose(0, 1).float()
        vh = v_b.transpose(0, 1).float()
        s = (qh @ kh.transpose(-1, -2)) * sm_scale
        if shape.causal:
            mask = torch.ones(Sq, Sk, dtype=torch.bool, device=t.q.device).tril(
                diagonal=Sk - Sq
            )
            s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = (p @ vh).transpose(0, 1).to(t.q.dtype)  # [Sq, H, D]
        outs.append(o)
    return torch.cat(outs, dim=0)


# =====================================================================
# Correctness
# =====================================================================
@dataclass
class CorrectnessResult:
    backend: str
    shape: str
    ok: bool
    max_abs: float = float("nan")
    mean_abs: float = float("nan")
    note: str = ""

    def short(self) -> str:
        if self.ok and "N/A" in self.note:
            return "N/A"
        if not self.ok and self.note:
            return f"FAIL ({self.note[:60]})"
        if not self.ok:
            return f"FAIL max={self.max_abs:.2e}"
        return f"ok({self.max_abs:.1e})"


def _tolerances(dtype, max_k, ref_kind):
    """
    Tolerances depend on what the reference is.

    * ref_kind == 'vllm_fa' (flash-style ref): both sides do the same online-
      softmax tricks, accumulate in fp32, and write back as fp16/bf16. The
      remaining differences come from (a) wgmma/cuBLAS vs Triton numerical
      order in the GEMMs and (b) tile-boundary rescale order. These are
      tight -- 1-2 ULP at fp16, a bit looser at bf16.

    * ref_kind == 'eager_fp32' (eager fp32 ref): the candidate accumulates
      in fp32 but the ref carries fp32 all the way through softmax, so a
      somewhat looser tolerance is needed.
    """
    if ref_kind == "vllm_fa":
        base_atol = 3e-3 if dtype == torch.bfloat16 else 1.5e-3
        base_rtol = 3e-3 if dtype == torch.bfloat16 else 1.5e-3
    else:  # eager_fp32
        base_atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
        base_rtol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    if max_k >= 8192:
        base_atol *= 2.0
        base_rtol *= 2.0
    return base_atol, base_rtol


def _compare(name, shape_name, out, ref, atol, rtol):
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(ref, tuple):
        ref = ref[0]
    if out.shape != ref.shape:
        return CorrectnessResult(
            name,
            shape_name,
            ok=False,
            note=f"shape mismatch {tuple(out.shape)} vs {tuple(ref.shape)}",
        )
    diff = (out.float() - ref.float()).abs()
    return CorrectnessResult(
        name,
        shape_name,
        ok=torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol),
        max_abs=diff.max().item(),
        mean_abs=diff.mean().item(),
    )


def _try_run(name, shape_name, runner, tensors, shape, ref, atol, rtol):
    try:
        with torch.inference_mode():
            out = runner(tensors, shape)
        return _compare(name, shape_name, out, ref, atol, rtol)
    except NotImplementedError as e:
        return CorrectnessResult(name, shape_name, ok=True, note=f"N/A ({e})")
    except Exception as e:
        return CorrectnessResult(
            name, shape_name, ok=False, note=f"{type(e).__name__}: {e}"
        )


def check_correctness(
    shape: Shape, t: Tensors, run_torch_check: bool, vllm_fa_version: int, ref_kind: str
):
    """
    ref_kind: 'vllm_fa' (preferred) or 'eager_fp32' (fallback).
    """
    results: List[CorrectnessResult] = []

    # Build the reference.
    if ref_kind == "vllm_fa":
        try:
            with torch.inference_mode():
                ref_raw = run_vllm_fa(t, shape, vllm_fa_version)
            ref = ref_raw[0] if isinstance(ref_raw, tuple) else ref_raw
            ref_note = f"reference (vllm-FA v{vllm_fa_version})"
        except NotImplementedError as e:
            # vLLM can't handle this shape -- fall back to eager for this row.
            try:
                with torch.inference_mode():
                    ref = eager_reference(t, shape)
                ref_kind = "eager_fp32"
                ref_note = f"reference (eager fp32; vllm N/A: {e})"
            except Exception as e2:
                results.append(
                    CorrectnessResult(
                        "ref",
                        shape.name,
                        ok=False,
                        note=f"both vllm+eager failed: {type(e2).__name__}: {e2}",
                    )
                )
                return None, results
        except Exception as e:
            results.append(
                CorrectnessResult(
                    "vllm_fa_ref", shape.name, ok=False, note=f"{type(e).__name__}: {e}"
                )
            )
            return None, results
    else:
        try:
            with torch.inference_mode():
                ref = eager_reference(t, shape)
            ref_note = "reference (eager fp32)"
        except Exception as e:
            results.append(
                CorrectnessResult(
                    "eager_ref", shape.name, ok=False, note=f"{type(e).__name__}: {e}"
                )
            )
            return None, results

    atol, rtol = _tolerances(t.q.dtype, t.max_seqlen_k, ref_kind)
    results.append(
        CorrectnessResult(
            "ref", shape.name, ok=True, max_abs=0.0, mean_abs=0.0, note=ref_note
        )
    )

    results.append(
        _try_run("flag_gems", shape.name, run_flag_gems, t, shape, ref, atol, rtol)
    )

    if run_torch_check:
        if shape.paged:
            results.append(
                CorrectnessResult(
                    "torch", shape.name, ok=True, note="N/A (no paged support)"
                )
            )
        else:
            results.append(
                _try_run(
                    "torch", shape.name, run_torch_varlen, t, shape, ref, atol, rtol
                )
            )

    return ref, results


# =====================================================================
# Bench
# =====================================================================
def attn_flops(shape: Shape) -> float:
    flops = 0.0
    for q_len, k_len in shape.seq_lens:
        gemm1 = 2.0 * q_len * k_len * shape.head_dim * shape.nh_q
        gemm2 = 2.0 * q_len * k_len * shape.head_dim * shape.nh_q
        flops += gemm1 + gemm2
    if shape.causal:
        flops *= 0.5
    return flops


def bench_one(fn, warmup_iters=10, bench_iters=50):
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(bench_iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    return times_ms[len(times_ms) // 2]


def fmt_tflops(flops, ms):
    if ms == 0 or ms != ms:
        return "  -    "
    return f"{flops / ms / 1e9:7.1f}"


def _try_bench(fn, warmup, iters):
    try:
        return bench_one(fn, warmup, iters)
    except NotImplementedError:
        return float("nan")
    except Exception:
        return float("nan")


def _print_ms_tflops(ms, flops):
    if ms != ms:
        print(f"{'N/A':>12}{'N/A':>12}", end="")
    else:
        print(f"{ms:>12.3f}{fmt_tflops(flops, ms):>12}", end="")


# =====================================================================
# Startup banner
# =====================================================================
def print_dispatch_banner():
    fn = flag_gems.flash_attn_varlen_func
    src = inspect.getsourcefile(fn) or "<unknown>"
    mod = getattr(fn, "__module__", "<unknown>")
    norm = src.replace("\\", "/")
    in_hopper = "runtime/backend/_nvidia/hopper/ops" in norm
    print(f"\nflag_gems.flash_attn_varlen_func")
    print(f"    __module__ = {mod}")
    print(f"    file       = {src}")
    if in_hopper:
        print(
            f"    [OK] Hopper override is installed -- benchmarking the migrated FA3 path."
        )
    else:
        print(f"    [WARN] NOT routed through hopper backend. You are benchmarking the")
        print(f"           generic flag_gems/ops/attention.py. Check that")
        print(f"           hopper/ops/__init__.py imports flash_attn_varlen_func.")
    print(f"    is_fa3_supported source: {_FA3_SUPPORT_FROM}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        choices=["prefill", "decode", "varlen", "paged", "all"],
        default="all",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--no-torch", action="store_true", help="Skip torch.nn.attention.varlen."
    )
    parser.add_argument(
        "--no-vllm-fa", action="store_true", help="Skip vllm.vllm_flash_attn."
    )
    parser.add_argument(
        "--vllm-fa-version",
        type=int,
        choices=[2, 3],
        default=3,
        help="fa_version kwarg passed to vllm flash_attn (default 3).",
    )
    parser.add_argument("--no-correctness", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda"

    if not args.no_torch and not HAS_TORCH_VARLEN:
        print("[WARN] torch.nn.attention.varlen.varlen_attn not available; skipping.")
        args.no_torch = True
    use_vllm_fa = (not args.no_vllm_fa) and HAS_VLLM_FA

    print_dispatch_banner()
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")
    print(f"FA3 supported: {is_fa3_supported()}")
    if HAS_TORCH_VARLEN:
        if TORCH_VARLEN_HAS_WINDOW:
            tv_mode = "window_size kwarg"
        elif TORCH_VARLEN_HAS_IS_CAUSAL:
            tv_mode = "is_causal kwarg"
        else:
            tv_mode = "no causal kwarg"
        scale_msg = (
            f", scale='{TORCH_VARLEN_SCALE_KWARG}'"
            if TORCH_VARLEN_SCALE_KWARG
            else ", no scale kwarg"
        )
        print(f"torch.nn.attention.varlen available: True ({tv_mode}{scale_msg})")
    else:
        print(f"torch.nn.attention.varlen available: False")
    if HAS_VLLM_FA:
        print(
            f"vllm.vllm_flash_attn available: True "
            f"(block_table={VLLM_FA_HAS_BLOCK_TABLE}, "
            f"seqused_k={VLLM_FA_HAS_SEQUSED_K}, "
            f"fa_version={VLLM_FA_HAS_FA_VERSION})"
        )
        if use_vllm_fa:
            print(f"  using fa_version={args.vllm_fa_version}")
    else:
        print(f"vllm.vllm_flash_attn available: False")
    print(f"Dtype: {args.dtype}, warmup={args.warmup}, iters={args.iters}")
    print(f"Correctness: {'OFF' if args.no_correctness else 'ON'}")

    if args.shapes == "prefill":
        shapes = prefill_shapes()
    elif args.shapes == "decode":
        shapes = decode_shapes()
    elif args.shapes == "varlen":
        shapes = varlen_mixed_shapes()
    elif args.shapes == "paged":
        shapes = paged_shapes()
    else:
        shapes = all_shapes()

    do_correctness = not args.no_correctness

    # Decide which reference to use for correctness.
    ref_kind = "vllm_fa" if (use_vllm_fa and HAS_VLLM_FA) else "eager_fp32"
    if do_correctness:
        if ref_kind == "vllm_fa":
            print(f"Correctness reference: vllm-FA (fa_version={args.vllm_fa_version})")
        else:
            if args.strict:
                print("[strict] vllm-FA unavailable; refusing to use eager fallback.")
                sys.exit(1)
            print(
                "Correctness reference: eager fp32 per-sequence SDPA "
                "(vllm-FA unavailable)"
            )
    print()

    # Header
    cols = ["Shape"]
    if do_correctness:
        cols += [f"correctness vs {'vllm' if ref_kind == 'vllm_fa' else 'eager'}"]
    cols += ["FlagGems ms", "FlagGems TFLOPS"]
    if not args.no_torch:
        cols += ["torch ms", "torch TFLOPS"]
    if use_vllm_fa:
        cols += ["vllm ms", "vllm TFLOPS"]
    if use_vllm_fa:
        cols += ["FG/vllm"]

    name_w = 44
    correctness_w = 48
    print(f"{'Shape':<{name_w}}", end="")
    for c in cols[1:]:
        w = correctness_w if c.startswith("correctness") else 14
        print(f"{c:>{w}}", end="")
    print()
    total_w = name_w + sum(
        correctness_w if c.startswith("correctness") else 14 for c in cols[1:]
    )
    print("-" * total_w)

    all_correctness: List[CorrectnessResult] = []
    for shape in shapes:
        flops = attn_flops(shape)
        t = make_varlen(shape, dtype, device)

        per_shape_results: List[CorrectnessResult] = []
        if do_correctness:
            _, per_shape_results = check_correctness(
                shape,
                t,
                run_torch_check=not args.no_torch,
                vllm_fa_version=args.vllm_fa_version,
                ref_kind=ref_kind,
            )
            all_correctness.extend(per_shape_results)

        correctness_summary = ""
        if do_correctness:
            parts = []
            for r in per_shape_results:
                if r.backend == "ref":
                    continue  # synthetic placeholder, not a tested backend
                parts.append(f"{r.backend}:{r.short()}")
            correctness_summary = " ".join(parts) if parts else "-"

        results: Dict[str, float] = {}

        fn = lambda t=t, shape=shape: run_flag_gems(t, shape)
        results["flag_gems"] = _try_bench(fn, args.warmup, args.iters)

        if not args.no_torch:
            if shape.paged:
                results["torch"] = float("nan")
            else:
                fn = lambda t=t, shape=shape: run_torch_varlen(t, shape)
                results["torch"] = _try_bench(fn, args.warmup, args.iters)

        if use_vllm_fa:
            fn = lambda t=t, shape=shape: run_vllm_fa(t, shape, args.vllm_fa_version)
            results["vllm"] = _try_bench(fn, args.warmup, args.iters)

        # Print row
        print(f"{shape.name:<{name_w}}", end="")
        if do_correctness:
            cs = correctness_summary
            if len(cs) > correctness_w - 1:
                cs = cs[: correctness_w - 2] + "…"
            print(f"{cs:>{correctness_w}}", end="")
        fg_ms = results.get("flag_gems", float("nan"))
        _print_ms_tflops(fg_ms, flops)
        if not args.no_torch:
            _print_ms_tflops(results.get("torch", float("nan")), flops)
        if use_vllm_fa:
            _print_ms_tflops(results.get("vllm", float("nan")), flops)
            v_ms = results.get("vllm", float("nan"))
            if v_ms == v_ms and fg_ms == fg_ms and fg_ms > 0:
                print(f"{v_ms/fg_ms:>13.2f}x", end="")
            else:
                print(f"{'-':>14}", end="")
        print()

    # Correctness summary
    if do_correctness:
        print()
        ref_label = (
            "vllm-FA" if ref_kind == "vllm_fa" else "eager fp32 per-sequence SDPA"
        )
        print("=" * 70)
        print(f"Correctness summary (reference = {ref_label})")
        print("=" * 70)
        backends: Dict[str, List[CorrectnessResult]] = {}
        for r in all_correctness:
            if r.backend == "ref":
                continue
            backends.setdefault(r.backend, []).append(r)
        any_fail = False
        for backend, rs in backends.items():
            n_pass = sum(1 for r in rs if r.ok and "N/A" not in r.note)
            n_na = sum(1 for r in rs if r.ok and "N/A" in r.note)
            n_fail = len(rs) - n_pass - n_na
            status = (
                f"PASS ({n_pass}/{len(rs)})"
                if n_fail == 0
                else f"FAIL ({n_fail}/{len(rs)})"
            )
            if n_na:
                status += f"  +{n_na} N/A"
            print(f"  {backend:<10} {status}")
            if n_fail:
                any_fail = True
                for r in rs:
                    if not r.ok:
                        line = f"      [{r.shape}]  {r.short()}"
                        if r.mean_abs == r.mean_abs:
                            line += f"  mean={r.mean_abs:.2e}"
                        print(line)
        print()
        if args.strict and any_fail:
            print("[strict] correctness failures detected, exiting non-zero.")
            sys.exit(1)


if __name__ == "__main__":
    main()
