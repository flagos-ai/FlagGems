import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

logger = logging.getLogger(__name__)


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 128}, num_warps=1),
    triton.Config({"BLOCK": 128}, num_warps=2),
    triton.Config({"BLOCK": 256}, num_warps=1),
    triton.Config({"BLOCK": 256}, num_warps=2),
    triton.Config({"BLOCK": 256}, num_warps=4),
    triton.Config({"BLOCK": 512}, num_warps=1),
    triton.Config({"BLOCK": 512}, num_warps=2),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=2),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=8),
    triton.Config({"BLOCK": 2048}, num_warps=2),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def _fake_quant_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale,
    zp_f,
    qmin_f,
    qmax_f,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # fp32 path exactly like ATen FakeQuant.cu
    inv_scale = 1.0 / scale
    xf = x.to(tl.float32) * inv_scale
    xq = libdevice.rint(xf) + zp_f
    # fmaxf / fminf semantics: NaN is replaced by the bound (NaN -> qmin)
    xq = tl.where(xq >= qmin_f, xq, qmin_f)
    xq = tl.where(xq <= qmax_f, xq, qmax_f)
    y = ((xq - zp_f) * scale).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _fake_quant_strided_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale,
    zp_f,
    qmin_f,
    qmax_f,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # decode flat logical index -> storage offset for an arbitrary (<=8 dim)
    # strided layout; unused trailing dims are padded with d=1, t=0.
    rem = offs
    src = (rem % d7) * t7
    rem = rem // d7
    src += (rem % d6) * t6
    rem = rem // d6
    src += (rem % d5) * t5
    rem = rem // d5
    src += (rem % d4) * t4
    rem = rem // d4
    src += (rem % d3) * t3
    rem = rem // d3
    src += (rem % d2) * t2
    rem = rem // d2
    src += (rem % d1) * t1
    rem = rem // d1
    src += (rem % d0) * t0
    x = tl.load(x_ptr + src, mask=mask, other=0.0)
    inv_scale = 1.0 / scale
    xf = x.to(tl.float32) * inv_scale
    xq = libdevice.rint(xf) + zp_f
    xq = tl.where(xq >= qmin_f, xq, qmin_f)
    xq = tl.where(xq <= qmax_f, xq, qmax_f)
    y = ((xq - zp_f) * scale).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, y, mask=mask)


def fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max):
    logger.debug("GEMS_ILUVATAR FAKE_QUANTIZE_PER_TENSOR_AFFINE")
    # aten qparam validation (reproduced verbatim in behaviour)
    qmin_i = int(quant_min)
    qmax_i = int(quant_max)
    zp_i = int(zero_point)
    if qmin_i > qmax_i:
        raise RuntimeError("`quant_min` should be less than or equal to `quant_max`.")
    if zp_i < qmin_i or zp_i > qmax_i:
        raise RuntimeError("`zero_point` must be between `quant_min` and `quant_max`.")

    x = input
    contiguous = x.is_contiguous()
    if contiguous:
        out = torch.empty_like(x)
    else:
        # memory_format matters: empty_like would mirror a strided input layout
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    n = x.numel()
    if n == 0:
        return out

    scale_f = float(scale)
    zp_f = float(zp_i)
    qmin_f = float(qmin_i)
    qmax_f = float(qmax_i)

    if contiguous:
        _fake_quant_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
            x,
            out,
            n,
            scale_f,
            zp_f,
            qmin_f,
            qmax_f,
        )
    else:
        shape = list(x.shape)
        strides = list(x.stride())
        ndim = len(shape)
        dims = [1] * 8
        sts = [0] * 8
        for i in range(ndim):
            dims[i] = shape[i]
            sts[i] = strides[i]
        BLOCK = 1024
        _fake_quant_strided_kernel[(triton.cdiv(n, BLOCK),)](
            x,
            out,
            n,
            scale_f,
            zp_f,
            qmin_f,
            qmax_f,
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            dims[4],
            dims[5],
            dims[6],
            dims[7],
            sts[0],
            sts[1],
            sts[2],
            sts[3],
            sts[4],
            sts[5],
            sts[6],
            sts[7],
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out
