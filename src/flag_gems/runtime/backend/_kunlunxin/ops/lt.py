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
_RAW_SMALL_SCALAR_LIMIT = 65536

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect(name="xpu3", file=os.path.join(_HERE, "lt_raw.xpu"))
    def lt_scalar_raw(
        in_, out, numel, esz, type_code, scalar_bits, chunk_start, chunk_count
    ): ...

    @triton.jit(
        do_not_specialize=["numel", "esz", "type_code", "scalar_bits", "chunk_count"]
    )
    def lt_scalar_raw_kernel(In, Out, numel, esz, type_code, scalar_bits, chunk_count):
        pid = tl.program_id(0)
        tle.raw.call(
            lt_scalar_raw,
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

    Matches torch.less's type promotion: the python float scalar is converted
    to the tensor's dtype and compared in that dtype. Cached: the dtype
    conversion costs a couple of microseconds on the host, which is directly
    visible on launch-bound small shapes.
    """
    if dtype == torch.float32:
        return int(torch.tensor(B, dtype=torch.float32).view(torch.int32).item())
    return int(torch.tensor(B, dtype=dtype).view(torch.int16).item())


def _raw_lt_scalar(A, B):
    """less(A, scalar) via the raw payload, or None when it does not apply.

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
        lt_scalar_raw_kernel[(_NCLUSTER,)](
            _view_u8(A), _view_u8(out), M, esz, type_code, s_bits, per
        )
    return out


logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    unroll_num=8,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


def lt(A, B):
    logger.debug("GEMS_KUNLUNXIN LT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = lt_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR")
    if A.numel() >= _RAW_SMALL_SCALAR_LIMIT:
        raw_out = _raw_lt_scalar(A, B)
        if raw_out is not None:
            return raw_out
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        if (
            numel >= _LT_SCALAR_FAST_TILE * _LT_SCALAR_MIN_GRID
            and numel % _LT_SCALAR_FAST_TILE == 0
        ):
            return _lt_scalar_fast(A, float(B), (numel // _LT_SCALAR_FAST_TILE,))
        if numel >= _LT_SCALAR_MASKED_MIN and numel % _LT_SCALAR_FAST_TILE != 0:
            return _lt_scalar_fast_masked(A, float(B), numel)
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        B = float(torch.tensor(float(B), dtype=dtype).item())
    res = lt_func_scalar(A, B)
    return res


_LT_SCALAR_FAST_TILE = 131072
_LT_SCALAR_MIN_GRID = 128
_LT_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def lt_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (scalar - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t)


def _lt_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    lt_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_LT_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def lt_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (scalar - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t, mask=mask)


def _lt_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _LT_SCALAR_FAST_TILE),)
    lt_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_LT_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def lt_func_(x, y):
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def lt_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_")
    if A.device != B.device:
        B = B.to(A.device)
    lt_func_(A, B, out0=A)
    return A


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def lt_func_scalar_(x, y):
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def lt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR_")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32)
        and numel >= _LT_SCALAR_INPLACE_FAST_TILE
        and numel % _LT_SCALAR_INPLACE_FAST_TILE == 0
        and numel // _LT_SCALAR_INPLACE_FAST_TILE >= _LT_SCALAR_INPLACE_MIN_GRID
        and float(B) == 0.0
    ):
        return _lt_scalar_inplace_fast(A)
    if A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        B = float(torch.tensor(float(B), dtype=A.dtype).item())
    lt_func_scalar_(A, B, out0=A)
    return A


_LT_SCALAR_INPLACE_FAST_TILE = 131072
_LT_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def lt_scalar_inplace_fast_kernel(x_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (0.0 - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _lt_scalar_inplace_fast(A):
    grid = (A.numel() // _LT_SCALAR_INPLACE_FAST_TILE,)
    lt_scalar_inplace_fast_kernel[grid](
        A,
        TILE=_LT_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


def less_(A, B):
    return lt_(A, B)


def less_scalar_(A, B):
    return lt_scalar_(A, B)
