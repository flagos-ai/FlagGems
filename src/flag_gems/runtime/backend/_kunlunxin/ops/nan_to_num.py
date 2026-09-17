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
_RAW_CHUNK_BYTES = 1792

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect(name="xpu3", file=os.path.join(_HERE, "nan_to_num_raw.xpu"))
    def nan_to_num_raw_(
        in_,
        out,
        numel,
        esz,
        type_code,
        nan_bits,
        pinf_bits,
        ninf_bits,
        chunk_start,
        chunk_count,
    ): ...

    @triton.jit(
        do_not_specialize=[
            "numel",
            "esz",
            "type_code",
            "nan_bits",
            "pinf_bits",
            "ninf_bits",
            "chunk_count",
        ]
    )
    def nan_to_num_raw_kernel(
        In, Out, numel, esz, type_code, nan_bits, pinf_bits, ninf_bits, chunk_count
    ):
        pid = tl.program_id(0)
        tle.raw.call(
            nan_to_num_raw_,
            (
                In,
                Out,
                numel,
                esz,
                type_code,
                nan_bits,
                pinf_bits,
                ninf_bits,
                pid * chunk_count,
                chunk_count,
            ),
        )


@functools.lru_cache(maxsize=1024)
def _replacement_bits(v, dtype):
    """A replacement scalar converted to `dtype`, as a sign-extended i32 bit
    pattern (fp16/bf16 in the low 16 bits) -- exactly torch's promotion of a
    python float to the tensor's dtype."""
    if dtype == torch.float32:
        return int(torch.tensor(v, dtype=torch.float32).view(torch.int32).item())
    return int(torch.tensor(v, dtype=dtype).view(torch.int16).item())


def _view_u8(t):
    """Byte view of a tensor."""
    return t.view(torch.uint8)


def _raw_nan_to_num_(A, nan, posinf, neginf):
    """In-place nan_to_num_ via the raw payload, or None when it does not
    apply (non-contiguous / unsupported dtype / empty / too large)."""
    if not _TLE_OK or not A.is_contiguous():
        return None
    type_code = _RAW_TYPE_CODE.get(A.dtype)
    if type_code is None:
        return None
    M = A.numel()
    if M == 0 or M > _RAW_MAX_ELEMS:
        return None
    esz = A.element_size()
    nb, pb, mb = (
        _replacement_bits(nan, A.dtype),
        _replacement_bits(posinf, A.dtype),
        _replacement_bits(neginf, A.dtype),
    )
    chunk_elems = _RAW_CHUNK_BYTES // esz
    total_chunks = (M + chunk_elems - 1) // chunk_elems
    per = (total_chunks + _NCLUSTER - 1) // _NCLUSTER
    with torch_device_fn.device(A.device):
        nan_to_num_raw_kernel[(_NCLUSTER,)](
            _view_u8(A), _view_u8(A), M, esz, type_code, nb, pb, mb, per
        )
    return A


config_ = CodeGenConfig(
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
    is_tensor=[True, False, False, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    x_bits = x.to(tl.float32).to(tl.int32, bitcast=True)
    x_nan = (x_bits & 0x7FFFFFFF) > 0x7F800000
    x_posinf = x_bits == 0x7F800000
    x_neginf = x_bits == 0xFF800000
    x = tl.where(x_nan, nan, x)
    x = tl.where(x_posinf, posinf, x)
    x = tl.where(x_neginf, neginf, x)
    return x


_RAW_MIN_ELEMS = 65536


def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_KUNLUNXIN NAN_TO_NUM")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    return nan_to_num_func(A, nan, posinf, neginf)


def nan_to_num_(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_KUNLUNXIN NAN_TO_NUM_")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    if A.numel() >= _RAW_MIN_ELEMS:
        raw_out = _raw_nan_to_num_(A, nan, posinf, neginf)
        if raw_out is not None:
            return raw_out
    return nan_to_num_func(A, nan, posinf, neginf, out0=A)
