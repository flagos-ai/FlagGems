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


@triton.jit
def _hc_head_fused_singlepass_kernel(
    residual_ptr,
    fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    out_ptr,
    H: tl.constexpr,
    rms_eps,
    hc_eps,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Single-pass per-token fusion: sqrsum + mix dot-products + weighted sum.

    Reads the residual exactly twice (once in the accumulation pass, once in the
    weighted-sum pass) and ``fn`` once, so it avoids the redundant fp32 upcast /
    standalone mm / intermediate tensors of the 4-kernel pipeline. HC is a
    constexpr in {2, 4} and each head is handled by a dedicated accumulator to
    keep the reduction simple on XPU (no dynamic indexing into an accumulator).

    ``fn`` loads are always in-bounds (the accumulation loop bounds h_off to
    [0, H) and every fn index ``k*HC*H + hc*H + h_off`` stays < K = HC*H), so
    they are unmasked regardless of NEED_MASK; only the residual loads/stores
    take the masked path when H is not a multiple of BLOCK_H.
    """
    pid = tl.program_id(0)
    x_base = pid * (HC * H)

    sqr_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix0 = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix1 = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix2 = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix3 = tl.zeros([BLOCK_H], dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h_off = h_start + tl.arange(0, BLOCK_H)

        if NEED_MASK:
            h_mask = h_off < H
            r0 = tl.load(
                residual_ptr + x_base + 0 * H + h_off, mask=h_mask, other=0.0
            ).to(tl.float32)
            r1 = tl.load(
                residual_ptr + x_base + 1 * H + h_off, mask=h_mask, other=0.0
            ).to(tl.float32)
        else:
            r0 = tl.load(residual_ptr + x_base + 0 * H + h_off).to(tl.float32)
            r1 = tl.load(residual_ptr + x_base + 1 * H + h_off).to(tl.float32)
        sqr_acc += r0 * r0 + r1 * r1

        mix0 += r0 * tl.load(fn_ptr + 0 * (HC * H) + 0 * H + h_off) + r1 * tl.load(
            fn_ptr + 0 * (HC * H) + 1 * H + h_off
        )
        mix1 += r0 * tl.load(fn_ptr + 1 * (HC * H) + 0 * H + h_off) + r1 * tl.load(
            fn_ptr + 1 * (HC * H) + 1 * H + h_off
        )

        if HC > 2:
            if NEED_MASK:
                r2 = tl.load(
                    residual_ptr + x_base + 2 * H + h_off, mask=h_mask, other=0.0
                ).to(tl.float32)
                r3 = tl.load(
                    residual_ptr + x_base + 3 * H + h_off, mask=h_mask, other=0.0
                ).to(tl.float32)
            else:
                r2 = tl.load(residual_ptr + x_base + 2 * H + h_off).to(tl.float32)
                r3 = tl.load(residual_ptr + x_base + 3 * H + h_off).to(tl.float32)
            sqr_acc += r2 * r2 + r3 * r3

            mix0 += r2 * tl.load(fn_ptr + 0 * (HC * H) + 2 * H + h_off) + r3 * tl.load(
                fn_ptr + 0 * (HC * H) + 3 * H + h_off
            )
            mix1 += r2 * tl.load(fn_ptr + 1 * (HC * H) + 2 * H + h_off) + r3 * tl.load(
                fn_ptr + 1 * (HC * H) + 3 * H + h_off
            )

            mix2 += r0 * tl.load(fn_ptr + 2 * (HC * H) + 0 * H + h_off) + r1 * tl.load(
                fn_ptr + 2 * (HC * H) + 1 * H + h_off
            )
            mix2 += r2 * tl.load(fn_ptr + 2 * (HC * H) + 2 * H + h_off) + r3 * tl.load(
                fn_ptr + 2 * (HC * H) + 3 * H + h_off
            )

            mix3 += r0 * tl.load(fn_ptr + 3 * (HC * H) + 0 * H + h_off) + r1 * tl.load(
                fn_ptr + 3 * (HC * H) + 1 * H + h_off
            )
            mix3 += r2 * tl.load(fn_ptr + 3 * (HC * H) + 2 * H + h_off) + r3 * tl.load(
                fn_ptr + 3 * (HC * H) + 3 * H + h_off
            )

    K = HC * H
    sqr_total = tl.sum(sqr_acc)
    rsqrt_val = tl.math.rsqrt(sqr_total / K + rms_eps)
    hc_scale = tl.load(hc_scale_ptr)

    mix0 = tl.sum(mix0)
    mix1 = tl.sum(mix1)
    pre_mix0 = (
        tl.sigmoid(mix0 * rsqrt_val * hc_scale + tl.load(hc_base_ptr + 0)) + hc_eps
    )
    pre_mix1 = (
        tl.sigmoid(mix1 * rsqrt_val * hc_scale + tl.load(hc_base_ptr + 1)) + hc_eps
    )

    if HC > 2:
        mix2 = tl.sum(mix2)
        mix3 = tl.sum(mix3)
        pre_mix2 = (
            tl.sigmoid(mix2 * rsqrt_val * hc_scale + tl.load(hc_base_ptr + 2)) + hc_eps
        )
        pre_mix3 = (
            tl.sigmoid(mix3 * rsqrt_val * hc_scale + tl.load(hc_base_ptr + 3)) + hc_eps
        )

    out_base = pid * H
    for h_start in range(0, H, BLOCK_H):
        h_off = h_start + tl.arange(0, BLOCK_H)
        if NEED_MASK:
            h_mask = h_off < H
            r0 = tl.load(
                residual_ptr + x_base + 0 * H + h_off, mask=h_mask, other=0.0
            ).to(tl.float32)
            r1 = tl.load(
                residual_ptr + x_base + 1 * H + h_off, mask=h_mask, other=0.0
            ).to(tl.float32)
        else:
            r0 = tl.load(residual_ptr + x_base + 0 * H + h_off).to(tl.float32)
            r1 = tl.load(residual_ptr + x_base + 1 * H + h_off).to(tl.float32)
        acc = pre_mix0 * r0 + pre_mix1 * r1
        if HC > 2:
            if NEED_MASK:
                r2 = tl.load(
                    residual_ptr + x_base + 2 * H + h_off, mask=h_mask, other=0.0
                ).to(tl.float32)
                r3 = tl.load(
                    residual_ptr + x_base + 3 * H + h_off, mask=h_mask, other=0.0
                ).to(tl.float32)
            else:
                r2 = tl.load(residual_ptr + x_base + 2 * H + h_off).to(tl.float32)
                r3 = tl.load(residual_ptr + x_base + 3 * H + h_off).to(tl.float32)
            acc += pre_mix2 * r2 + pre_mix3 * r3
        if NEED_MASK:
            tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16), mask=h_mask)
        else:
            tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16))


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
    fn_c = fn.contiguous()
    out_c = out if out.is_contiguous() else torch.empty_like(out)

    if HC in (2, 4):
        # Single-pass fusion: reads residual twice + fn once, no fp32 upcast of
        # the whole [N, K] tensor and no standalone mm. `isCloseUnrollControl`
        # skips the TritonXPUUnrollControl pass that overflows uni_sram on the
        # large unrolled HC=4 body (same workaround as linalg_householder_product
        # / bincount / rms_norm_backward).
        block_h = 1024
        _hc_head_fused_singlepass_kernel[(num_tokens,)](
            residual_c,
            fn_c,
            hc_scale,
            hc_base,
            out_c,
            H=H,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            HC=HC,
            BLOCK_H=block_h,
            NEED_MASK=(H % block_h != 0),
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        if out.data_ptr() != out_c.data_ptr():
            out.copy_(out_c)
        return out

    # Fallback for HC > 4: the original 4-kernel pipeline.
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

    mixes = _gems_mm(x2d.to(torch.float32), fn_c.t())

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
