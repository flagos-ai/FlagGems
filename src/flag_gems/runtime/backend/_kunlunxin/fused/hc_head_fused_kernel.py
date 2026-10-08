import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.fused.mhc.hc_head_fused_kernel import (
    hc_head_fused_kernel as _general_hc_head_fused_kernel,
)

logger = logging.getLogger(__name__)

_PART_BLOCK = {2: 512, 4: 1024}
_ROW_BLOCK_MAX = 8192
_T_MAX = 128


@triton.jit
def _sqrsum_partials_kernel(
    residual_ptr,
    part_ptr,
    columns,
    B: tl.constexpr,
    T: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    """Exact per-token tile squares (grid (N, cdiv(K, B)))."""
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs = pid_t * B + tl.arange(0, B)
    if ALIGNED:
        v = tl.load(residual_ptr + pid_n * columns + offs).to(tl.float32)
    else:
        v = tl.load(
            residual_ptr + pid_n * columns + offs, mask=offs < columns, other=0.0
        ).to(tl.float32)
    tl.store(part_ptr + pid_n * T + pid_t, tl.sum(v * v))


@triton.jit
def _head_mix_kernel(
    part_ptr,
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_mix_ptr,
    T: tl.constexpr,
    K: tl.constexpr,
    rms_eps,
    hc_eps,
    HC: tl.constexpr,
):
    """Per-token rsqrt + sigmoid (any HC), all scalar loads, no reductions."""
    pid = tl.program_id(0)
    sq = 0.0
    for i in tl.static_range(T):
        sq += tl.load(part_ptr + pid * T + i)
    rms_inv = tl.rsqrt(sq / K + rms_eps)
    hc_scale = tl.load(hc_scale_ptr)
    for k in tl.static_range(HC):
        m = tl.load(mixes_ptr + pid * HC + k)
        b = tl.load(hc_base_ptr + k)
        p = tl.sigmoid(m * rms_inv * hc_scale + b) + hc_eps
        tl.store(pre_mix_ptr + pid * HC + k, p)


@triton.jit
def _weighted_row_kernel(
    residual_ptr,
    pre_mix_ptr,
    out_ptr,
    H: tl.constexpr,
    HC: tl.constexpr,
    B: tl.constexpr,
):
    """Weighted sum of the HC residual rows (masked, no reduction)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    m = offs < H
    base = pid * (HC * H)
    acc = tl.zeros([B], dtype=tl.float32)
    for k in tl.static_range(HC):
        pk = tl.load(pre_mix_ptr + pid * HC + k)
        r = tl.load(residual_ptr + base + k * H + offs, mask=m, other=0.0).to(
            tl.float32
        )
        acc += pk * r
    tl.store(out_ptr + pid * H + offs, acc.to(tl.bfloat16), mask=m)


def hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    """HC head fused kernel (kunlunxin / XPU specialized)."""
    assert hs_flat.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    num_tokens = hs_flat.shape[0]
    if num_tokens == 0:
        return out

    assert hs_flat.shape == (num_tokens, hc_mult, hidden_size)
    assert fn.shape == (hc_mult, hc_mult * hidden_size)
    assert hc_scale.shape == (1,)
    assert hc_base.shape == (hc_mult,)
    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == hs_flat.dtype

    if hs_flat.device.type != "cuda":
        return _general_hc_head_fused_kernel(
            hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
        )

    H = hidden_size
    HC = hc_mult
    K = HC * H
    B = _PART_BLOCK.get(HC, 512)
    T = (K + B - 1) // B
    if T > _T_MAX:
        return _general_hc_head_fused_kernel(
            hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
        )

    residual_c = hs_flat.contiguous()
    out_c = out if out.is_contiguous() else torch.empty_like(out)

    x2d = residual_c.reshape(num_tokens, K)

    part = torch.empty(num_tokens, T, dtype=torch.float32, device=hs_flat.device)
    _sqrsum_partials_kernel[(num_tokens, T)](
        x2d,
        part,
        K,
        B=B,
        T=T,
        ALIGNED=(K % B == 0),
        num_warps=4,
        num_stages=1,
    )

    from flag_gems.runtime.backend._kunlunxin.ops.mm import mm as _gems_mm

    mixes = _gems_mm(x2d.to(torch.float32), fn.t())

    pre_mix = torch.empty(num_tokens, HC, dtype=torch.float32, device=hs_flat.device)
    _head_mix_kernel[(num_tokens,)](
        part,
        mixes,
        hc_scale,
        hc_base,
        pre_mix,
        T=T,
        K=K,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        HC=HC,
        num_warps=4,
        num_stages=1,
    )

    row_block = min(triton.next_power_of_2(H), _ROW_BLOCK_MAX)
    _weighted_row_kernel[(num_tokens,)](
        residual_c,
        pre_mix,
        out_c,
        H=H,
        HC=HC,
        B=row_block,
        num_warps=8,
        num_stages=1,
    )

    if out.data_ptr() != out_c.data_ptr():
        out.copy_(out_c)
    return out


def _use_general_for_ab():
    """A/B escape hatch (FLAGGEMS_XPU_HC_HEAD_GENERAL=1 forces the general impl)."""
    return os.environ.get("FLAGGEMS_XPU_HC_HEAD_GENERAL", "0") == "1"


def _install():
    """Find and replace the mhc entrypoint attribute (direct-import family)."""
    if _use_general_for_ab():
        return
    import sys

    mod = sys.modules.get("flag_gems.fused.mhc.hc_head_fused_kernel")
    if mod is not None:
        cur = getattr(mod, "hc_head_fused_kernel", None)
        if cur is _general_hc_head_fused_kernel:
            mod.hc_head_fused_kernel = hc_head_fused_kernel


def _ensure_vllm_collect_works():
    """Neutralize the vllm site-packages import-hook collection crash (XPU env)."""
    try:
        import importlib

        importlib.import_module("vllm_xpu._C")

        return
    except Exception:
        pass
    import sys
    import types

    for _name in ("vllm_xpu._C", "vllm_xpu._moe_C"):
        if _name not in sys.modules:
            _m = types.ModuleType(_name)
            _m.__file__ = f"<seeded by flag_gems {__name__}>"
            sys.modules[_name] = _m


_install()
_ensure_vllm_collect_works()
