import functools
import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCLUSTER = 12
_RAW_MAX_ELEMS = 2**31 - 1
_RAW_CHUNK_BYTES = 2048
_SMALL_SCALAR_LIMIT = 65536

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect(name="xpu3", file=os.path.join(_HERE, "ne_raw.xpu"))
    def ne_scalar_raw(
        in_, out, numel, esz, type_code, scalar_bits, chunk_start, chunk_count
    ): ...

    @triton.jit(
        do_not_specialize=["numel", "esz", "type_code", "scalar_bits", "chunk_count"]
    )
    def ne_scalar_raw_kernel(In, Out, numel, esz, type_code, scalar_bits, chunk_count):
        pid = tl.program_id(0)
        tle.raw.call(
            ne_scalar_raw,
            (
                In,
                Out,
                numel,
                esz,
                type_code,
                scalar_bits,
                pid * chunk_count,
                chunk_count,
            ),
        )


def _view_u8(t):
    """Byte view of a tensor; works for 0-dim tensors too."""
    if t.dim() == 0:
        return t.view(1).view(torch.uint8)
    return t.view(torch.uint8)


@functools.lru_cache(maxsize=1024)
def _scalar_bits(B, dtype):
    """The scalar promoted to `dtype`, as a sign-extended int32 bit pattern.

    Matches torch.not_equal's type promotion: the python float scalar is
    converted to the tensor's dtype and compared in that dtype. Cached: the
    dtype conversion costs a couple of microseconds on the host, which is
    directly visible on launch-bound small shapes.
    """
    if dtype == torch.float32:
        return int(torch.tensor(B, dtype=torch.float32).view(torch.int32).item())
    return int(torch.tensor(B, dtype=dtype).view(torch.int16).item())


def _raw_not_equal_scalar(A, B):
    """not_equal(A, scalar) via the raw payload, or None when it does not apply.

    Only the contiguous case in the supported float dtypes is handled;
    anything else falls back to the pointwise scalar kernel below.
    """
    if not _TLE_OK or not A.is_contiguous():
        return None
    type_code = _RAW_TYPE_CODE.get(A.dtype)
    if type_code is None:
        return None
    M = A.numel()
    if M == 0 or M > _RAW_MAX_ELEMS:
        return None
    esz = A.element_size()
    s_bits = _scalar_bits(B, A.dtype)
    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    chunk_elems = _RAW_CHUNK_BYTES // esz
    total_chunks = (M + chunk_elems - 1) // chunk_elems
    per = (total_chunks + _NCLUSTER - 1) // _NCLUSTER
    with torch_device_fn.device(A.device):
        ne_scalar_raw_kernel[(_NCLUSTER,)](
            _view_u8(A), _view_u8(out), M, esz, type_code, s_bits, per
        )
    return out


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
    buffer_size_limit=4096,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def not_equal_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def not_equal(A, B):
    logger.debug("GEMS_KUNLUNXIN NOT_EQUAL")
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.dtype == B.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and 0 < numel <= _NOT_EQUAL_TENSOR_FAST_MAX
    ):
        if numel <= _NOT_EQUAL_TENSOR_SMALL_MAX:
            return _not_equal_tensor_fast(A, B, numel, _NOT_EQUAL_TENSOR_TILE_SMALL)
        return _not_equal_tensor_fast(A, B, numel, _NOT_EQUAL_TENSOR_TILE_MID)
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = not_equal_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


_NOT_EQUAL_TENSOR_TILE_SMALL = 2048
_NOT_EQUAL_TENSOR_SMALL_MAX = 16384
_NOT_EQUAL_TENSOR_TILE_MID = 8192
_NOT_EQUAL_TENSOR_FAST_MAX = 65536


@triton.jit
def not_equal_tensor_fast_kernel(x_ptr, y_ptr, out_ptr, n_elements, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * TILE + tl.arange(0, TILE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offset, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offset, x != y, mask=mask)


@triton.jit
def not_equal_tensor_fast_unmasked_kernel(x_ptr, y_ptr, out_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = tl.load(y_ptr + offset).to(tl.float32)
    tl.store(out_ptr + offset, x != y)


def _not_equal_tensor_fast(A, B, numel, TILE):
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    out = torch.empty_like(A, dtype=torch.bool)
    try:
        if numel % TILE == 0:
            not_equal_tensor_fast_unmasked_kernel[(numel // TILE,)](
                A,
                B,
                out,
                TILE=TILE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
        else:
            not_equal_tensor_fast_kernel[(triton.cdiv(numel, TILE),)](
                A,
                B,
                out,
                numel,
                TILE=TILE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
        return out
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def not_equal_func_scalar(x, y):
    return x.to(tl.float32) != y


def not_equal_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN NOT_EQUAL_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and numel >= _NOT_EQUAL_SCALAR_MASKED_MIN
    ):
        s = float(B)
        wrapped = torch.tensor(s, dtype=dtype).item()
        if math.isfinite(wrapped):
            if (
                numel % _NOT_EQUAL_SCALAR_FAST_TILE == 0
                and numel >= _NOT_EQUAL_SCALAR_FAST_TILE * _NOT_EQUAL_SCALAR_MIN_GRID
            ):
                return _not_equal_scalar_fast(
                    A, float(wrapped), (numel // _NOT_EQUAL_SCALAR_FAST_TILE,)
                )
            if numel % _NOT_EQUAL_SCALAR_FAST_TILE != 0:
                return _not_equal_scalar_fast_masked(A, float(wrapped), numel)
    res = not_equal_func_scalar(A, B)
    return res


_NOT_EQUAL_SCALAR_FAST_TILE = 131072
_NOT_EQUAL_SCALAR_MIN_GRID = 128
_NOT_EQUAL_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def not_equal_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    d = tl.abs(x - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t)


def _not_equal_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    not_equal_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_NOT_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    return out32.to(torch.bool)


@triton.jit
def not_equal_scalar_fast_masked_kernel(
    out_ptr, y_ptr, scalar, numel, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    y = tl.load(y_ptr + tid, mask=mask).to(tl.float32)
    d = tl.abs(y - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t, mask=mask)


def _not_equal_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _NOT_EQUAL_SCALAR_FAST_TILE),)
    not_equal_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_NOT_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    return out32.to(torch.bool)
