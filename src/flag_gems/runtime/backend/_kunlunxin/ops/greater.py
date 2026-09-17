import functools
import logging
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
_RAW_SMALL_SCALAR_LIMIT = 65536

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect(name="xpu3", file=os.path.join(_HERE, "gt_raw.xpu"))
    def gt_scalar_raw(
        in_, out, numel, esz, type_code, scalar_bits, chunk_start, chunk_count
    ): ...

    @triton.jit(
        do_not_specialize=["numel", "esz", "type_code", "scalar_bits", "chunk_count"]
    )
    def gt_scalar_raw_kernel(In, Out, numel, esz, type_code, scalar_bits, chunk_count):
        pid = tl.program_id(0)
        tle.raw.call(
            gt_scalar_raw,
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

    Matches torch.greater's type promotion: the python float scalar is
    converted to the tensor's dtype and compared in that dtype. Cached: the
    dtype conversion costs a couple of microseconds on the host, which is
    directly visible on launch-bound small shapes.
    """
    if dtype == torch.float32:
        return int(torch.tensor(B, dtype=torch.float32).view(torch.int32).item())
    return int(torch.tensor(B, dtype=dtype).view(torch.int16).item())


def _raw_greater_scalar(A, B):
    """greater(A, scalar) via the raw payload, or None when it does not apply.

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
        gt_scalar_raw_kernel[(_NCLUSTER,)](
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
)


config_scalar = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=16,
    buffer_size_limit=8192,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def greater_func(x, y):
    return x.to(tl.float32) > y


def greater(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = greater_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


def greater_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_OUT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    if out is None:
        res = greater_func(A, B)
    else:
        greater_func(A, B, out0=out)
        res = out
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_scalar,
)
@triton.jit
def greater_func_scalar(x, y):
    return x.to(tl.float32) > y


def greater_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR")
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and (numel := A.numel()) >= _GREATER_SCALAR_FAST_TILE
        and numel % _GREATER_SCALAR_FAST_TILE == 0
        and numel // _GREATER_SCALAR_FAST_TILE >= _GREATER_SCALAR_MIN_GRID
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        return _greater_scalar_fast(A, float(B))
    res = greater_func_scalar(A, B)
    return res


_GREATER_SCALAR_FAST_TILE = 131072
_GREATER_SCALAR_MIN_GRID = 512


@triton.jit
def greater_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (x - scalar) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t)


def _greater_scalar_fast(A, scalar):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (A.numel() // _GREATER_SCALAR_FAST_TILE,)
    greater_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_GREATER_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


def _greater_scalar_out_fast(A, scalar, out):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (A.numel() // _GREATER_SCALAR_FAST_TILE,)
    greater_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_GREATER_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    torch.ops.aten._copy_from(out32, out, False)
    return out


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR_OUT")
    if (
        out is not None
        and A.is_contiguous()
        and out.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and (numel := A.numel()) >= _GREATER_SCALAR_FAST_TILE
        and numel % _GREATER_SCALAR_FAST_TILE == 0
        and numel // _GREATER_SCALAR_FAST_TILE >= _GREATER_SCALAR_MIN_GRID
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        return _greater_scalar_out_fast(A, float(B), out)
    if out is None:
        res = greater_func_scalar(A, B)
    else:
        greater_func_scalar(A, B, out0=out)
        res = out
    return res
