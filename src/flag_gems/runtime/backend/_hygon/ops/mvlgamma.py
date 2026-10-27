import torch
import triton
import triton.language as tl

MAX_DIMS = 16
_LARGE_N = 1 << 20  # grid-stride + wide config above this many elements
_HUGE_N = 1 << 26  # extra program-cap boost for the biggest tensors
_GS_BLOCK_LARGE = 1024
_GS_BLOCK_HUGE = 1024
_GS_BLOCK_SMALL = 128
_NPROG_CAP = 8192
_NPROG_HUGE = 16384


@triton.jit
def _lg(x):
    return tl.extra.libdevice.lgamma(x)


@triton.jit
def _mvlgamma_gs_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    P: tl.constexpr,
    BLOCK: tl.constexpr,
    CAST_F32: tl.constexpr,
    USE_I64: tl.constexpr,
    NEED_MASK: tl.constexpr,
    NDIM: tl.constexpr,
    shape_ptr,
    strides_ptr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for base0 in range(pid * BLOCK, n_elements, nprog * BLOCK):
        if USE_I64:
            offs = base0.to(tl.int64) + tl.arange(0, BLOCK).to(tl.int64)
        else:
            offs = base0 + tl.arange(0, BLOCK)
        if NEED_MASK:
            mask = offs < n_elements
        if NDIM == 1:
            if NEED_MASK:
                x = tl.load(x_ptr + offs, mask=mask, other=1.0)
            else:
                x = tl.load(x_ptr + offs)
        else:
            rem = offs
            in_idx = tl.zeros([BLOCK], dtype=tl.int64)
            for d in tl.static_range(NDIM):
                dim = tl.load(shape_ptr + d)
                c = rem % dim
                rem = rem // dim
                in_idx += c * tl.load(strides_ptr + d)
            if NEED_MASK:
                x = tl.load(x_ptr + in_idx, mask=mask, other=1.0)
            else:
                x = tl.load(x_ptr + in_idx)
        if CAST_F32:
            xc = x.to(tl.float32)
        else:
            xc = x
        acc = tl.zeros_like(xc)
        if P <= 32:
            for k in tl.static_range(P):
                acc += _lg(xc - 0.5 * k)
        else:
            for k in range(0, P):
                acc += _lg(xc - 0.5 * k)
        res = acc + (P * (P - 1) * 0.25 * 1.1447298858494002)
        if CAST_F32:
            res = res.to(x.dtype)
        if NEED_MASK:
            tl.store(out_ptr + offs, res, mask=mask)
        else:
            tl.store(out_ptr + offs, res)


def run(self, p):
    if isinstance(p, torch.Tensor):
        p = p.item()
    p = int(p)
    if p < 1:
        raise ValueError("mvlgamma: p must be a positive integer")

    out = torch.empty_like(self)
    n_elements = self.numel()
    if n_elements == 0:
        return out

    dtype = self.dtype
    cast_f32 = (dtype == torch.float16) or (dtype == torch.bfloat16)

    if n_elements >= _LARGE_N:
        if n_elements >= _HUGE_N:
            block = _GS_BLOCK_HUGE
            nprog = min(triton.cdiv(n_elements, block), _NPROG_HUGE)
        else:
            block = _GS_BLOCK_LARGE
            nprog = min(triton.cdiv(n_elements, block), _NPROG_CAP)
        num_warps = 8
    else:
        block = _GS_BLOCK_SMALL
        num_warps = 4
        nprog = min(triton.cdiv(n_elements, block), _NPROG_CAP)
    use_i64 = n_elements >= (1 << 31)
    need_mask = (n_elements % (nprog * block)) != 0

    if self.is_contiguous():
        _mvlgamma_gs_kernel[(nprog,)](
            self,
            out,
            n_elements,
            P=p,
            BLOCK=block,
            CAST_F32=cast_f32,
            USE_I64=use_i64,
            NEED_MASK=need_mask,
            NDIM=1,
            shape_ptr=out,
            strides_ptr=out,
            num_warps=num_warps,
        )
    else:
        shape = self.shape
        ndim = len(shape)
        shape_t = torch.tensor(
            list(shape) + [1] * (MAX_DIMS - ndim), dtype=torch.int64, device=self.device
        )
        strides_t = torch.tensor(
            list(self.stride()) + [0] * (MAX_DIMS - ndim),
            dtype=torch.int64,
            device=self.device,
        )
        _mvlgamma_gs_kernel[(nprog,)](
            self,
            out,
            n_elements,
            P=p,
            BLOCK=block,
            CAST_F32=cast_f32,
            USE_I64=use_i64,
            NEED_MASK=need_mask,
            NDIM=ndim,
            shape_ptr=shape_t,
            strides_ptr=strides_t,
            num_warps=num_warps,
        )
    return out


# Alias for FlagGems import convention
mvlgamma = run
