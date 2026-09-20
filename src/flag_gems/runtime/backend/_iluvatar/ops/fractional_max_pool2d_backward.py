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


@triton.jit
def _zero_kernel(out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    tl.store(
        out_ptr + offs, tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty), mask=mask
    )


@triton.jit
def _gather_kernel(
    grad_ptr,
    ind_ptr,
    out_ptr,
    W_out,
    M_in,
    W_in: tl.constexpr,
    HW_in: tl.constexpr,
    HW_out: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Exact-partition fast path (H_in == KH*H_out and W_in == KW*W_out): the
    # pooling windows tile the input exactly, so every input position belongs to
    # exactly one output window. out[pos] = grad[window(pos)] if
    # indices[window(pos)] == pos else 0. Purely coalesced, no atomics, no zero
    # pass, one kernel launch. W_in/HW_in/HW_out are constexpr so the index
    # divisions become magic-multiply sequences (measured ~10% faster on 16-bit
    # and ~5% on fp32 s3 vs runtime divisions).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M_in
    plane = offs // HW_in
    pos = offs - plane * HW_in
    h = pos // W_in
    w = pos - h * W_in
    oh = h // KH
    ow = w // KW
    src = plane * HW_out + oh * W_out + ow
    idx = tl.load(ind_ptr + src.to(tl.int64) * 2, mask=mask, other=0)
    g = tl.load(grad_ptr + src, mask=mask, other=0.0)
    val = tl.where(idx == pos, g, tl.zeros([BLOCK], dtype=g.dtype))
    tl.store(out_ptr + offs, val, mask=mask)


@triton.jit
def _window_gather_kernel(
    grad_ptr,
    ind_ptr,
    out_ptr,
    W_out,
    W_in,
    HW_in,
    HW_out,
    M_out,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Output-window-centric exact-partition gather (fp32 fast path): one lane per
    # output window loads indices/grad once (4x fewer index/grad reads than the
    # input-side gather), then writes the kh*kw child positions (coalesced across
    # lanes), storing 0 for non-matching children. Measured ~3-4% faster than the
    # input-side gather for fp32 on this target.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M_out
    plane = offs // HW_out
    rem = offs - plane * HW_out
    oh = rem // W_out
    ow = rem - oh * W_out
    idx = tl.load(ind_ptr + offs.to(tl.int64) * 2, mask=mask, other=-1)
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    base = (
        plane.to(tl.int64) * HW_in.to(tl.int64)
        + (oh.to(tl.int64) * KH) * W_in.to(tl.int64)
        + ow.to(tl.int64) * KW
    )
    for dh in tl.static_range(KH):
        for dw in tl.static_range(KW):
            child = base + dh * W_in.to(tl.int64) + dw
            hit = idx == (child - plane.to(tl.int64) * HW_in.to(tl.int64))
            val = tl.where(hit, g, tl.zeros([BLOCK], dtype=g.dtype))
            tl.store(out_ptr + child, val, mask=mask)


@triton.jit
def _scatter_kernel_flat(
    grad_ptr,
    ind_ptr,
    out_ptr,
    HW_out,
    HW_in,
    M,
    BLOCK: tl.constexpr,
    ATOMIC: tl.constexpr,
    IND32: tl.constexpr,
    RELAXED: tl.constexpr,
):
    # Flat 1D grid over all M = planes * HW_out output elements (contiguous inputs).
    # plane is recovered with a per-element integer division; this keeps the block
    # count proportional to total work even when HW_out is tiny (e.g. 9) and the
    # plane count is huge (e.g. 65536), avoiding thousands of idle blocks.
    # ATOMIC=True uses atomic_add (correct for overlapping pooling windows);
    # ATOMIC=False uses a plain store (correct when pooling windows are disjoint,
    # which guarantees the per-plane max indices are unique).
    # IND32=True: indices are viewed as int32 (little-endian low word at 2*offs),
    # halving index read traffic. RELAXED uses relaxed-memory-order atomics.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    plane = offs // HW_out
    if IND32:
        idx = tl.load(ind_ptr + offs.to(tl.int64) * 2, mask=mask, other=0)
    else:
        idx = tl.load(ind_ptr + offs, mask=mask, other=0)
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    dst = plane.to(tl.int64) * HW_in.to(tl.int64) + idx.to(tl.int64)
    if ATOMIC:
        if RELAXED:
            tl.atomic_add(out_ptr + dst, g, mask=mask, sem="relaxed")
        else:
            tl.atomic_add(out_ptr + dst, g, mask=mask)
    else:
        tl.store(out_ptr + dst, g, mask=mask)


@triton.jit
def _scatter_kernel_4d(
    grad_ptr,
    ind_ptr,
    out_ptr,
    gs_n,
    gs_c,
    gs_h,
    gs_w,
    is_n,
    is_c,
    is_h,
    is_w,
    os_n,
    os_c,
    os_h,
    os_w,
    W_out,
    W_in,
    HW_out,
    BLOCK: tl.constexpr,
):
    # General path: arbitrary strides, grid (blocks along HW_out, C, N).
    pid0 = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW_out

    oh = offs // W_out
    ow = offs - oh * W_out
    base_g = n.to(tl.int64) * gs_n + c.to(tl.int64) * gs_c
    base_i = n.to(tl.int64) * is_n + c.to(tl.int64) * is_c
    gsrc = base_g + oh.to(tl.int64) * gs_h + ow.to(tl.int64) * gs_w
    isrc = base_i + oh.to(tl.int64) * is_h + ow.to(tl.int64) * is_w
    idx = tl.load(ind_ptr + isrc, mask=mask, other=0)
    g = tl.load(grad_ptr + gsrc, mask=mask, other=0.0)
    i = idx.to(tl.int64) // W_in
    j = idx.to(tl.int64) - i * W_in
    dst = n.to(tl.int64) * os_n + c.to(tl.int64) * os_c + i * os_h + j * os_w
    tl.atomic_add(out_ptr + dst, g, mask=mask)


def fractional_max_pool2d_backward(
    grad_output, input, kernel_size, output_size, indices
):
    if input.numel() == 0 or grad_output.numel() == 0:
        return torch.empty_like(input)

    N, C, H_in, W_in = input.shape
    planes = N * C
    HW_in = H_in * W_in
    H_out, W_out = grad_output.shape[-2], grad_output.shape[-1]
    HW_out = H_out * W_out

    # Parse kernel_size (int or pair). With kernel (kh,kw) and output ~ input/k,
    # pooling windows are disjoint and per-plane max indices are unique, so a
    # plain store reproduces the scatter-add exactly; otherwise use atomics.
    # On this target, fp32 scattered plain stores are ~2x slower than fp32
    # atomics, while fp16/bf16 scattered stores are ~25% faster than atomics,
    # so the store path is restricted to the 16-bit dtypes (measured).
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size
    disjoint = (H_in >= kh * H_out) and (W_in >= kw * W_out)
    use_store = disjoint and (grad_output.dtype in (torch.float16, torch.bfloat16))
    contiguous = (
        input.is_contiguous()
        and grad_output.is_contiguous()
        and indices.is_contiguous()
    )

    # Exact-partition gather path: one coalesced kernel, no atomics, no zero pass.
    # fp32 uses the output-window-centric variant (measured ~3-4% faster, loads
    # idx/grad once per window); 16-bit dtypes use the input-side variant with
    # constexpr (magic-multiply) division, which matches or beats the store-scatter
    # on all exact shapes and removes the zero-kernel launch.
    exact = (HW_out > 0) and (H_in == kh * H_out) and (W_in == kw * W_out)
    if contiguous and exact:
        out = torch.empty_like(input)
        ind_arg = indices.view(torch.int32) if indices.dtype == torch.int64 else indices
        if grad_output.dtype == torch.float32:
            M_out = planes * HW_out
            _window_gather_kernel[(triton.cdiv(M_out, 128),)](
                grad_output,
                ind_arg,
                out,
                W_out,
                W_in,
                HW_in,
                HW_out,
                M_out,
                KH=int(kh),
                KW=int(kw),
                BLOCK=128,
                num_warps=2,
            )
        else:
            M_in = planes * HW_in
            _gather_kernel[(triton.cdiv(M_in, 1024),)](
                grad_output,
                ind_arg,
                out,
                W_out,
                M_in,
                W_in=W_in,
                HW_in=HW_in,
                HW_out=HW_out,
                KH=int(kh),
                KW=int(kw),
                BLOCK=1024,
                num_warps=8,
            )
        return out

    out = torch.empty_like(input)
    total = planes * HW_in
    ZB = 4096
    _zero_kernel[(triton.cdiv(total, ZB),)](out, total, BLOCK=ZB, num_warps=4)

    if HW_out == 0:
        return out

    BLOCK = 128
    NW = 2
    M = planes * HW_out
    if M == 0:
        return out
    if out.is_contiguous() and grad_output.is_contiguous() and indices.is_contiguous():
        use_ind32 = indices.dtype == torch.int64
        if use_ind32:
            ind_arg = indices.view(torch.int32)
        else:
            ind_arg = indices
        _scatter_kernel_flat[(triton.cdiv(M, BLOCK),)](
            grad_output,
            ind_arg,
            out,
            HW_out,
            HW_in,
            M,
            BLOCK=BLOCK,
            ATOMIC=not use_store,
            IND32=use_ind32,
            RELAXED=not use_store,
            num_warps=NW,
        )
    else:
        gs = grad_output.stride()
        iss = indices.stride()
        os = out.stride()
        x = triton.cdiv(HW_out, 1024)
        y = min(C, 65535)
        z = triton.cdiv(N, 65535)
        _scatter_kernel_4d[(x, y, z)](
            grad_output,
            indices,
            out,
            gs[0],
            gs[1],
            gs[2],
            gs[3],
            iss[0],
            iss[1],
            iss[2],
            iss[3],
            os[0],
            os[1],
            os[2],
            os[3],
            W_out,
            W_in,
            HW_out,
            BLOCK=1024,
            num_warps=4,
        )
    return out
