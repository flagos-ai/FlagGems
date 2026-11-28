# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Exact mHC post followed by an explicit, already-known pre combine.

Both BF16 materialization boundaries and the original FP32 FMA order are
preserved. The caller supplies predecessor pre coefficients explicitly.
"""

import functools
import hashlib
from pathlib import Path

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
except ModuleNotFoundError:
    tle = None


@triton.jit
def _row(values, INDEX: tl.constexpr, BLOCK: tl.constexpr):
    return tl.gather(values, tl.full((1, BLOCK), INDEX, tl.int32), axis=0).reshape(
        (BLOCK,)
    )


@triton.jit
def _post_combine_values(x, r, POST, COMB, PRE, row, BLOCK: tl.constexpr):
    channels = tl.arange(0, 4)
    post = tl.load(POST + row * 4 + channels).to(tl.float32)
    comb0 = tl.load(COMB + row * 16 + channels).to(tl.float32)
    r0 = _row(r, 0, BLOCK)
    # This is the existing exact TileLang/NVCC contraction contract.
    first_product = comb0[:, None] * r0[None, :]
    values = tl.fma(post[:, None], x[None, :], first_product)
    for source in tl.static_range(1, 4):
        coefficient = tl.load(COMB + row * 16 + source * 4 + channels).to(tl.float32)
        residual = _row(r, source, BLOCK)
        values = tl.fma(coefficient[:, None], residual[None, :], values)
    post_bf16 = values.to(tl.bfloat16)
    post_fp32 = post_bf16.to(tl.float32)
    pre0 = tl.load(PRE + row * 4).to(tl.float32)
    v0 = _row(post_fp32, 0, BLOCK)
    # FlagTree 3.6 baseline PTX starts with fma.rn.f32(..., +0).
    combined = tl.fma(pre0, v0, tl.full((BLOCK,), 0.0, tl.float32))
    for source in tl.static_range(1, 4):
        coefficient = tl.load(PRE + row * 4 + source).to(tl.float32)
        value = _row(post_fp32, source, BLOCK)
        combined = tl.fma(coefficient, value, combined)
    return post_bf16, combined.to(tl.bfloat16)


@triton.jit
def _post_combine_kernel(
    X,
    R,
    POST,
    COMB,
    PRE,
    Y,
    Z,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_TLE: tl.constexpr,
    SERIAL_H: tl.constexpr,
):
    row = tl.program_id(0)
    hr = tl.arange(0, BLOCK)
    channels = tl.arange(0, 4)
    if USE_TLE:
        xb = tle.gpu.alloc(
            [BLOCK],
            dtype=tl.bfloat16,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        rb = tle.gpu.alloc(
            [4, BLOCK],
            dtype=tl.bfloat16,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        xp = tle.gpu.local_ptr(xb, (hr,))
        cr = tl.broadcast_to(channels[:, None], (4, BLOCK))
        hc = tl.broadcast_to(hr[None, :], (4, BLOCK))
        rp = tle.gpu.local_ptr(rb, (cr, hc))
    iterations: tl.constexpr = triton.cdiv(H, BLOCK) if SERIAL_H else 1
    for iteration in range(iterations):
        base = iteration * BLOCK if SERIAL_H else tl.program_id(1) * BLOCK
        h = base + hr
        xm = h < H
        xptr = X + row * H + h
        rptr = R + (row * 4 + channels[:, None]) * H + h[None, :]
        if USE_TLE:
            tle.gpu.copy(xptr, xb, [BLOCK], mask=xm)
            tle.gpu.copy(rptr, rb, [4, BLOCK], mask=xm[None, :])
            x = tl.load(xp).to(tl.float32)
            r = tl.load(rp).to(tl.float32)
        else:
            x = tl.load(xptr, xm, 0.0).to(tl.float32)
            r = tl.load(rptr, xm[None, :], 0.0).to(tl.float32)
        post_out, combined = _post_combine_values(x, r, POST, COMB, PRE, row, BLOCK)
        tl.store(
            Y + (row * 4 + channels[:, None]) * H + h[None, :], post_out, xm[None, :]
        )
        tl.store(Z + row * H + h, combined, xm)


@triton.jit
def _post_combine_stream_kernel(
    X,
    R,
    POST,
    COMB,
    PRE,
    Y,
    Z,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_TLE: tl.constexpr,
    SERIAL_H: tl.constexpr,
):
    row = tl.program_id(0)
    hr = tl.arange(0, BLOCK)
    if USE_TLE:
        xb = tle.gpu.alloc(
            [BLOCK],
            dtype=tl.bfloat16,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        rb = tle.gpu.alloc(
            [4, BLOCK],
            dtype=tl.bfloat16,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        xp = tle.gpu.local_ptr(xb, (hr,))
        rp0 = tle.gpu.local_ptr(rb, (tl.full((BLOCK,), 0, tl.int32), hr))
        rp1 = tle.gpu.local_ptr(rb, (tl.full((BLOCK,), 1, tl.int32), hr))
        rp2 = tle.gpu.local_ptr(rb, (tl.full((BLOCK,), 2, tl.int32), hr))
        rp3 = tle.gpu.local_ptr(rb, (tl.full((BLOCK,), 3, tl.int32), hr))
        channels = tl.arange(0, 4)
    iterations: tl.constexpr = triton.cdiv(H, BLOCK) if SERIAL_H else 1
    for iteration in range(iterations):
        base = iteration * BLOCK if SERIAL_H else tl.program_id(1) * BLOCK
        h = base + hr
        mask = h < H
        if USE_TLE:
            tle.gpu.copy(X + row * H + h, xb, [BLOCK], mask=mask)
            tle.gpu.copy(
                R + (row * 4 + channels[:, None]) * H + h[None, :],
                rb,
                [4, BLOCK],
                mask=mask[None, :],
            )
            x = tl.load(xp).to(tl.float32)
            r0 = tl.load(rp0).to(tl.float32)
            r1 = tl.load(rp1).to(tl.float32)
            r2 = tl.load(rp2).to(tl.float32)
            r3 = tl.load(rp3).to(tl.float32)
        else:
            x = tl.load(X + row * H + h, mask, 0.0).to(tl.float32)
            r0 = tl.load(R + (row * 4) * H + h, mask, 0.0).to(tl.float32)
            r1 = tl.load(R + (row * 4 + 1) * H + h, mask, 0.0).to(tl.float32)
            r2 = tl.load(R + (row * 4 + 2) * H + h, mask, 0.0).to(tl.float32)
            r3 = tl.load(R + (row * 4 + 3) * H + h, mask, 0.0).to(tl.float32)
        combined = tl.full((BLOCK,), 0.0, tl.float32)
        for target in tl.static_range(4):
            post = tl.load(POST + row * 4 + target)
            c0 = tl.load(COMB + row * 16 + target)
            c1 = tl.load(COMB + row * 16 + 4 + target)
            c2 = tl.load(COMB + row * 16 + 8 + target)
            c3 = tl.load(COMB + row * 16 + 12 + target)
            value = tl.fma(post, x, c0 * r0)
            value = tl.fma(c1, r1, value)
            value = tl.fma(c2, r2, value)
            value = tl.fma(c3, r3, value)
            rounded = value.to(tl.bfloat16)
            coefficient = tl.load(PRE + row * 4 + target)
            combined = tl.fma(coefficient, rounded.to(tl.float32), combined)
            tl.store(Y + (row * 4 + target) * H + h, rounded, mask)
        tl.store(Z + row * H + h, combined.to(tl.bfloat16), mask)


def post_combine_candidate(
    x,
    residual,
    post,
    comb,
    apply_pre,
    *,
    use_tle=True,
    block=1024,
    serial_h=True,
    warps=4,
    stages=2,
    out_post=None,
    out_combined=None,
    streamed=False,
):
    """Return (BF16 post residual, BF16 combined input), with explicit variant.

    This research entry has no auto-install, fallback, or hidden state. The
    final measured dispatch will be provided only after real-value validation.
    """
    m, h = x.shape
    assert m > 0 and h == 5120 and residual.shape == (m, 4, h)
    assert post.shape in ((m, 4), (m, 4, 1))
    assert comb.shape == (m, 4, 4) and apply_pre.shape == (m, 4)
    assert x.dtype == residual.dtype == torch.bfloat16
    assert post.dtype == comb.dtype == apply_pre.dtype == torch.float32
    assert all(
        t.device == x.device and t.is_contiguous()
        for t in (x, residual, post, comb, apply_pre)
    )
    assert x.is_cuda
    y = torch.empty_like(residual) if out_post is None else out_post
    z = torch.empty_like(x) if out_combined is None else out_combined
    assert y.shape == residual.shape and z.shape == x.shape
    assert y.dtype == z.dtype == torch.bfloat16
    assert y.device == z.device == x.device and y.is_contiguous() and z.is_contiguous()
    inputs = {
        t.untyped_storage().data_ptr() for t in (x, residual, post, comb, apply_pre)
    }
    assert (
        y.untyped_storage().data_ptr() not in inputs
        and z.untyped_storage().data_ptr() not in inputs
    )
    assert y.untyped_storage().data_ptr() != z.untyped_storage().data_ptr()
    if m:
        kernel = _post_combine_stream_kernel if streamed else _post_combine_kernel
        kernel[(m, 1 if serial_h else triton.cdiv(h, block))](
            x,
            residual,
            post,
            comb,
            apply_pre,
            y,
            z,
            H=h,
            BLOCK=block,
            USE_TLE=use_tle,
            SERIAL_H=serial_h,
            num_warps=warps,
            num_stages=stages,
            enable_fp_fusion=False,
        )
    return y, z


def torch_reference(x, residual, post, comb, apply_pre):
    """Independent literal FP32 semantic reference, with both BF16 boundaries.

    Eager Torch operations need not contract the same way as the CUDA baseline;
    baseline natural error is recorded separately from the exact primary gate.
    """
    m, h = x.shape
    post = post.reshape(m, 4)
    value = post[:, :, None] * x.float()[:, None, :]
    for source in range(4):
        value = value + comb[:, source, :, None] * residual[:, source, None, :].float()
    post_bf16 = value.bfloat16()
    combined = torch.zeros((m, h), dtype=torch.float32, device=x.device)
    for source in range(4):
        combined = (
            combined + apply_pre[:, source, None] * post_bf16[:, source, :].float()
        )
    return post_bf16, combined.bfloat16()


MHC_FUSION_ABI = 1
SUPPORTED_M = (2048, 4096, 8192, 16384, 32768)
EXPECTED_LIBTRITON_SHA256 = (
    "fd75320edd31ad2fa38b76f7e16d6512990f3447d28b11981ea546668dce1e83"
)


@functools.lru_cache(maxsize=1)
def stack_identity():
    path = Path(triton.__file__).parent / "_C" / "libtriton.so"
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "triton": triton.__version__.split("+")[0],
        "libtriton_sha256": sha,
        "abi": MHC_FUSION_ABI,
    }


@functools.lru_cache(maxsize=32)
def _device_supported(index):
    return torch.cuda.get_device_capability(index) == (
        9,
        0,
    ) and "H100" in torch.cuda.get_device_name(index)


def supports(x, residual, post, comb, apply_pre):
    """Metadata-only dispatch guard; no tensor-value scan or synchronization."""
    tensors = (x, residual, post, comb, apply_pre)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    if not (
        x.is_cuda and x.ndim == 2 and x.shape[0] in SUPPORTED_M and x.shape[1] == 5120
    ):
        return False
    m = x.shape[0]
    identity = stack_identity()
    return (
        residual.shape == (m, 4, 5120)
        and post.shape == (m, 4)
        and comb.shape == (m, 4, 4)
        and apply_pre.shape == (m, 4)
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == apply_pre.dtype == torch.float32
        and all(
            t.device == x.device and t.is_contiguous() and t.data_ptr() % 16 == 0
            for t in tensors
        )
        and identity["triton"] == "3.6.0"
        and identity["libtriton_sha256"] == EXPECTED_LIBTRITON_SHA256
        and _device_supported(x.device.index)
    )


def post_combine(
    x, residual, post, comb, apply_pre, *, out_post=None, out_combined=None
):
    """Public exact operator for the measured domain; caller owns fallback.

    Returns fresh BF16 post/combined tensors by default, never mutating inputs.
    Optional outputs must be disjoint BF16 contiguous buffers. A model adapter
    must provide the already-known predecessor pre and call its exact original
    path whenever supports() is false.
    """
    if not supports(x, residual, post, comb, apply_pre):
        raise ValueError(
            "post_combine called outside the verified H100/FlagTree3.6 domain"
        )
    return post_combine_candidate(
        x,
        residual,
        post,
        comb,
        apply_pre,
        use_tle=False,
        streamed=True,
        block=1024,
        serial_h=True,
        warps=4,
        stages=1,
        out_post=out_post,
        out_combined=out_combined,
    )
