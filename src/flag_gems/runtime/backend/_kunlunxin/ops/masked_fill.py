import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, "NO_OPMATH")],
    config=_config,
)
@triton.jit
def masked_fill_kernel(inp, expand_mask, value):
    return tl.where(expand_mask, value, inp)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, "NO_OPMATH")],
    config=_config,
)
@triton.jit
def masked_fill_tensor_value_kernel(inp, expand_mask, value):
    return tl.where(expand_mask, value, inp)


_FAST_TILE = 131072
_FAST_MIN_NUMEL = 1 << 20
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def masked_fill_fast_kernel(
    out_ptr, x_ptr, mask_ptr, V: tl.constexpr, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    xi = tl.load(x_ptr + tid)
    m = tl.load(mask_ptr + tid).to(xi.dtype)
    r = xi + (V - xi) * m
    tl.store(out_ptr + tid, r)


@triton.jit
def masked_fill_fast_masked_kernel(
    out_ptr, x_ptr, mask_ptr, numel, V: tl.constexpr, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    m0 = tid < numel
    xi = tl.load(x_ptr + tid, mask=m0)
    m = tl.load(mask_ptr + tid, mask=m0).to(xi.dtype)
    r = xi + (V - xi) * m
    tl.store(out_ptr + tid, r, mask=m0)


def _fast_bits(value, dtype):
    iview = torch.int16 if dtype in (torch.float16, torch.bfloat16) else torch.int32
    return int(torch.tensor([value], dtype=dtype).view(iview).item())


def _masked_fill_fast(inp, mask, value, out):
    n = inp.numel()
    bits = _fast_bits(value, inp.dtype)
    xi = inp.view(
        torch.int16 if inp.dtype in (torch.float16, torch.bfloat16) else torch.int32
    )
    oi = out.view(xi.dtype)
    mask8 = mask.view(torch.int8)
    launch = dict(
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    if n % _FAST_TILE == 0:
        masked_fill_fast_kernel[(n // _FAST_TILE,)](
            oi, xi, mask8, V=bits, TILE=_FAST_TILE, **launch
        )
    else:
        masked_fill_fast_masked_kernel[(math.ceil(n / _FAST_TILE),)](
            oi, xi, mask8, n, V=bits, TILE=_FAST_TILE, **launch
        )
    return out


try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCLUSTER = 12
_RAW_MAX_ELEMS = 2**31 - 1
_RAW_CHUNK_BYTES = 1280

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "masked_fill_raw.xpu"))
    def masked_fill_raw_(
        in_, mask, numel, esz, type_code, value_bits, chunk_start, chunk_count
    ): ...

    @triton.jit(
        do_not_specialize=[
            "numel",
            "esz",
            "type_code",
            "value_bits",
            "chunk_count",
        ]
    )
    def masked_fill_raw_kernel(
        In, Mask, numel, esz, type_code, value_bits, chunk_count
    ):
        pid = tl.program_id(0)
        tle.raw.call(
            masked_fill_raw_,
            (
                In,
                Mask,
                numel,
                esz,
                type_code,
                value_bits,
                pid * chunk_count,
                chunk_count,
            ),
        )


def _raw_masked_fill_(inp, mask, value):
    """In-place masked_fill_ via the raw payload, or None when it does not
    apply (non-contiguous / broadcast mask / unsupported dtype / empty)."""
    if not _TLE_OK or not inp.is_contiguous() or not mask.is_contiguous():
        return None
    type_code = _RAW_TYPE_CODE.get(inp.dtype)
    if type_code is None:
        return None
    if tuple(mask.shape) != tuple(inp.shape):
        return None
    M = inp.numel()
    if M == 0 or M > _RAW_MAX_ELEMS:
        return None
    value_bits = _fast_bits(value, inp.dtype)
    esz = inp.element_size()
    chunk_elems = _RAW_CHUNK_BYTES // esz
    total_chunks = (M + chunk_elems - 1) // chunk_elems
    per = (total_chunks + _NCLUSTER - 1) // _NCLUSTER
    with torch_device_fn.device(inp.device):
        masked_fill_raw_kernel[(_NCLUSTER,)](
            inp.view(torch.uint8),
            mask.view(torch.uint8),
            M,
            esz,
            type_code,
            value_bits,
            per,
        )
    return inp


_RAW_MIN_ELEMS = 65536


def _use_fast_path(inp, mask, value):
    if torch.is_tensor(value):
        return False
    if inp.dtype not in _FAST_DTYPES:
        return False
    if not (inp.is_contiguous() and mask.is_contiguous()):
        return False
    if tuple(mask.shape) != tuple(inp.shape):
        return False
    return inp.numel() >= _FAST_MIN_NUMEL


def masked_fill(inp, mask, value):
    logger.debug("GEMS_KUNLUNXIN MASKED_FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        if value.device != inp.device:
            raise RuntimeError("masked_fill value must be on the input device")
        kernel = masked_fill_tensor_value_kernel
    else:
        kernel = masked_fill_kernel
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        out = torch.empty_like(inp)
        kernel(inp, mask, value, out0=out)
        return out

    out = torch.empty_like(inp, dtype=inp.dtype, device=inp.device)
    if inp.numel() == 0:
        return out

    if _use_fast_path(inp, mask, value):
        return _masked_fill_fast(inp, mask, value, out)

    if inp.is_contiguous() and tuple(mask.shape) == tuple(inp.shape):
        mask = mask.contiguous()
        kernel(inp.view(-1), mask.view(-1), value, out0=out.view(-1))
    else:
        expand_mask = mask.expand(inp.shape)
        kernel.instantiate(inp.ndim)
        kernel(inp, expand_mask, value, out0=out)
    return out


def masked_fill_(inp, mask, value):
    logger.debug("GEMS_KUNLUNXIN MASKED_FILL_")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        if value.device != inp.device:
            raise RuntimeError("masked_fill value must be on the input device")
        kernel = masked_fill_tensor_value_kernel
    else:
        kernel = masked_fill_kernel
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        kernel(inp, mask, value, out0=inp)
        return inp

    if inp.numel() == 0:
        return inp

    if _use_fast_path(inp, mask, value):
        if inp.numel() >= _RAW_MIN_ELEMS:
            raw_out = _raw_masked_fill_(inp, mask, value)
            if raw_out is not None:
                return raw_out
        return _masked_fill_fast(inp, mask, value, inp)

    if inp.is_contiguous() and tuple(mask.shape) == tuple(inp.shape):
        mask = mask.contiguous()
        kernel(inp.view(-1), mask.view(-1), value, out0=inp.view(-1))
    else:
        expand_mask = mask.expand(inp.shape)
        kernel.instantiate(inp.ndim)
        kernel(inp, expand_mask, value, out0=inp)
    return inp
