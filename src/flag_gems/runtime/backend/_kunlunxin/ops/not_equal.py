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

# ---------------------------------------------------------------------------
# tle.raw fast path for the large-shape tensor-vs-scalar compare (P800 xpu3,
# cluster C payload in ne_raw.xpu).
#
# The compiler cannot vectorize a float compare whose second operand is a
# scalar: CoreTiling hands the scalar kernel 8 elements/core/iteration (below
# the f32 vector width of 16), so `arith.cmpf` stays scalarized (~600us on 16M
# elements). The only compiler fast path is a tensor-vs-tensor compare, which
# materializes a full-size broadcast tensor (extra 32MB write + read -> 82us,
# speedup ~0.43). The hand-written payload drives per-core GM2LM/LM2GM DMA and
# compares against the scalar with the hardware svneq_* vector intrinsics,
# reading the input once (32MB fp16/bf16 or 64MB fp32) and writing the 1
# byte/element bool result (16MB) -- the same footprint as the ATen reference,
# see ne_raw.xpu for the full analysis. See op-opti/max_dim.md for the original
# payload recipe this mirrors.
try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCLUSTER = 12  # P800 (xpu3): one Triton program == one cluster of 64 cores
# Payload scalars are i32 (do_not_specialize); guard the byte range.
_RAW_MAX_ELEMS = 2**31 - 1
# Must match CHUNK_BYTES in ne_raw.xpu (the chunk-grid partition contract).
_RAW_CHUNK_BYTES = 2048

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "ne_raw.xpu"))
    def ne_scalar_raw(in_, out, numel, esz, type_code, scalar_bits,
                      chunk_start, chunk_count):
        ...

    @triton.jit(do_not_specialize=["numel", "esz", "type_code", "scalar_bits",
                                   "chunk_count"])
    def ne_scalar_raw_kernel(In, Out, numel, esz, type_code, scalar_bits,
                             chunk_count):
        pid = tl.program_id(0)
        tle.raw.call(ne_scalar_raw, (In, Out, numel, esz, type_code,
                                     scalar_bits, pid * chunk_count,
                                     chunk_count))


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
    # fp16 / bf16: the payload only reads the low 16 bits.
    return int(torch.tensor(B, dtype=dtype).view(torch.int16).item())


def _raw_not_equal_scalar(A, B):
    """not_equal(A, scalar) via the raw payload, or None when it does not apply.

    Only the contiguous full-size case is handled (benchmark shapes are
    contiguous); non-contiguous inputs fall back to the hybrid path below.
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
    # partition by payload chunks (CHUNK_BYTES/esz elements each): every
    # program and core gets whole chunks so all GM2LM/LM2GM transfers are
    # CHUNK_BYTES-aligned in global memory.
    chunk_elems = _RAW_CHUNK_BYTES // esz
    total_chunks = (M + chunk_elems - 1) // chunk_elems
    per = (total_chunks + _NCLUSTER - 1) // _NCLUSTER
    with torch_device_fn.device(A.device):
        ne_scalar_raw_kernel[(_NCLUSTER,)](
            _view_u8(A), _view_u8(out), M, esz, type_code, s_bits, per)
    return out


# NOTE: `not_equal` / `not_equal_scalar` are aliases of `ne` / `ne_scalar`.
# kunlunxin overrides `ne` with the tuned config below but previously left
# `not_equal` UNCOVERED, so it fell to the generic `ops/not_equal.py` bare
# `@pointwise_dynamic` (no CodeGenConfig, no kunlunAutoGrid / unroll_num) and was
# stuck at the launch-bound / narrow-DMA baseline (IR
# `harness/perf_ir_3/ir-not_equal-dev6.log`). Mirroring the sibling `ne` recipe
# verbatim (tuned config + kunlunAutoGrid + unroll_num) lifts throughput with
# zero algorithm change.
#
# `buffer_size_limit=4096` bounds the per-core DMA tile (same lever proven on
# acos/isfinite). On the large benchmark shapes (268M / 65536-wide) it shaves a
# consistent ~4% off fp16/bf16 and ~10% off fp32 gems latency (fp32 268M
# 1.853->1.661ms, fp32 65536-wide 4.474->4.006ms) with no change on small
# shapes; the default launch path used buffer_size_limit=2048.
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
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = not_equal_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def not_equal_func_scalar(x, y):
    return x.to(tl.float32) != y


# Below this element count the operation is launch-bound and the bare scalar
# kernel (single launch, no intermediate tensor) measures slightly better
# through the aten dispatch than the payload path (whose host side pays for
# the output allocation and chunk-grid arithmetic), so keep it for small
# shapes; the payload wins from ~64K elements up.
_SMALL_SCALAR_LIMIT = 65536


def not_equal_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN NOT_EQUAL_SCALAR")
    # Fast path: the hand-written tle.raw payload (ne_raw.xpu). The compiler
    # cannot vectorize a tensor-vs-scalar float compare on XPU (CoreTiling
    # gives the pointwise scalar kernel 8 elements/core/iteration, below the
    # f32 vector width of 16, so `arith.cmpf` stays scalarized -- ~600us on
    # 16M elements), while the payload streams A once with per-core pipelined
    # GM2LM/LM2GM DMA and the hardware vector-ne intrinsics, writing the
    # 1 byte/element bool result with the same memory footprint as ATen. It
    # also matches ATen's scalar-dtype promotion exactly (the bare scalar
    # kernel compares in f32, which diverges from ATen for non-exact scalars).
    if A.numel() >= _SMALL_SCALAR_LIMIT:
        raw_out = _raw_not_equal_scalar(A, B)
        if raw_out is not None:
            return raw_out

    # Fallback (small shapes, unsupported dtypes, non-contiguous inputs):
    # the bare scalar kernel is a single launch and fine. It must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST (fp16 tensor-vs-scalar
    # compare trips `arith.cmpf same-type` / uni_sram overflow -> compile
    # failure).
    if A.numel() < _SMALL_SCALAR_LIMIT:
        return not_equal_func_scalar(A, B)

    # Above it, a tensor-vs-tensor compare is the only compiler fast path:
    # materialize the scalar as a contiguous full-size broadcast tensor
    # (torch.full_like, ~24us for 16M fp16) and route through the tuned
    # tensor-tensor `not_equal_func` with the fusion env vars (~62us for 16M
    # fp16) -> ~86us total vs ~462us, i.e. ~5x.  Semantics are preserved:
    # torch.not_equal NaN and +/-0.0 handling is identical for tensor-vs-scalar
    # and tensor-vs-tensor.
    B_broadcast = torch.full_like(A, B)
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        res = not_equal_func(A, B_broadcast)
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return res
