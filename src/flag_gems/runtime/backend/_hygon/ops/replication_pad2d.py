import torch
import triton
import triton.language as tl


# Primary fast path: flat gather over the whole output.  W_out/H_out are
# compile-time constants so the per-element divmods lower to magic-number
# sequences (measured ~1.3-1.7x faster than runtime divisors on the medium
# and large benchmark shapes; the ALU was ~half the kernel time).
@triton.jit
def _replication_pad2d_flat_cx(
    in_ptr,
    out_ptr,
    TOTAL,
    H_in,
    W_in,
    top,
    left,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    ow = offs % W_out
    t = offs // W_out
    oh = t % H_out
    img = t // H_out
    ih = tl.minimum(tl.maximum(oh - top, 0), H_in - 1)
    iw = tl.minimum(tl.maximum(ow - left, 0), W_in - 1)
    HWi = H_in.to(tl.int64) * W_in.to(tl.int64)
    in_off = (
        img.to(tl.int64) * HWi + ih.to(tl.int64) * W_in.to(tl.int64) + iw.to(tl.int64)
    )
    # Uniform branch: only the last CTA can be partially covered, so full CTAs
    # run unmasked loads/stores (small measured gain on the large shapes).
    if (pid + 1) * BLOCK <= TOTAL:
        v = tl.load(in_ptr + in_off)
        tl.store(out_ptr + offs.to(tl.int64), v)
    else:
        m = offs < TOTAL
        v = tl.load(in_ptr + in_off, mask=m)
        tl.store(out_ptr + offs.to(tl.int64), v, mask=m)


# Generic fallback for shapes that do not match the constexpr specialization.
@triton.jit
def _replication_pad2d_flat_rt(
    in_ptr,
    out_ptr,
    TOTAL,
    H_in,
    W_in,
    H_out,
    W_out,
    top,
    left,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < TOTAL
    ow = offs % W_out
    t = offs // W_out
    oh = t % H_out
    img = t // H_out
    ih = tl.minimum(tl.maximum(oh - top, 0), H_in - 1)
    iw = tl.minimum(tl.maximum(ow - left, 0), W_in - 1)
    HWi = H_in.to(tl.int64) * W_in.to(tl.int64)
    in_off = (
        img.to(tl.int64) * HWi + ih.to(tl.int64) * W_in.to(tl.int64) + iw.to(tl.int64)
    )
    v = tl.load(in_ptr + in_off, mask=m)
    tl.store(out_ptr + offs.to(tl.int64), v, mask=m)


def _parse_padding(padding):
    # Fast path: the harness passes a 4-element list/tuple.
    if type(padding) in (list, tuple) and len(padding) == 4:
        return (
            int(padding[0]),
            int(padding[1]),
            int(padding[2]),
            int(padding[3]),
        )
    if isinstance(padding, torch.Tensor):
        vals = padding.detach().cpu().tolist()
    else:
        vals = padding
    try:
        seq = tuple(int(v) for v in vals)
    except TypeError:
        seq = (int(vals), int(vals), int(vals), int(vals))
    if len(seq) == 1:
        seq = (seq[0],) * 4
    elif len(seq) == 2:
        seq = (seq[0], seq[1], 0, 0)
    if len(seq) != 4:
        raise ValueError("replication_pad2d expects padding of length 4")
    return seq  # (left, right, top, bottom)


def run(input, padding):
    if not input.is_contiguous():
        input = input.contiguous()

    left, right, top, bottom = _parse_padding(padding)

    if input.ndim < 2:
        raise ValueError("replication_pad2d requires at least 2D input")

    *leading, H_in, W_in = input.shape
    NC = 1
    for s in leading:
        NC *= s
    H_out = H_in + top + bottom
    W_out = W_in + left + right

    out = torch.empty((*leading, H_out, W_out), dtype=input.dtype, device=input.device)

    if NC == 0 or H_out <= 0 or W_out <= 0:
        return out

    total = NC * H_out * W_out

    if W_out <= 128 and H_out <= 128 and total >= 100_000:
        grid = (triton.cdiv(total, 1024),)
        # Verified: num_warps=8 is ~12% faster than 4 for fp32 widths near 64
        # (the 56x56-padded workload); other geometries prefer 4.
        nw = 8 if (input.dtype == torch.float32 and 56 <= W_out <= 64) else 4
        _replication_pad2d_flat_cx[grid](
            input,
            out,
            total,
            H_in,
            W_in,
            top,
            left,
            H_out=H_out,
            W_out=W_out,
            BLOCK=1024,
            num_warps=nw,
        )
    else:
        grid = (triton.cdiv(total, 2048),)
        _replication_pad2d_flat_rt[grid](
            input,
            out,
            total,
            H_in,
            W_in,
            H_out,
            W_out,
            top,
            left,
            BLOCK=2048,
            num_warps=4,
        )
    return out


# Alias for FlagGems import convention
replication_pad2d = run
