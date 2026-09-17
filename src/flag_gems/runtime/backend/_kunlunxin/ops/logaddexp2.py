import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Two independent defects fixed here relative to the generic ops/logaddexp2.py:
#
# 1) PERFORMANCE — config-less pointwise_dynamic recompiles per shape.
#    The generic op decorates logaddexp2_func with @pointwise_dynamic WITHOUT a
#    config, so on XPU the kernel is specialized per shape
#    (logaddexp2_func__fp16S512S_...), triggering per-shape recompilation and an
#    IR explosion (see harness/perf_ir_4/ir-logaddexp2-dev1.log, 84 modules).
#    Every gm2lm/lm2gm degrades to the discrete path -> large shapes ran at
#    ~0.011 speedup (70-2900ms). Fix = pass the standard XPU CodeGenConfig
#    (kunlunAutoGrid=True, prefer_1d_tile, unroll_num=8) so the kernel is
#    shape-independent, compiled once, and does contiguous block DMA. This is
#    the same config used by cos/acos/logical_and.
#
# 2) CORRECTNESS — tl.exp2 / tl.log2 are natural-base on this XPU.
#    Isolation (/tmp/exp2log2_iso.py) proved that on this backend
#    `tl.exp2(x)` returns e**x (NOT 2**x) and `tl.log2(z)` returns ln(z)
#    (NOT log2(z)). The generic formula m + log2(1 + 2**(-|d|)) therefore
#    computed m + ln(1 + e**(-|d|)), wrong by an ln(2) factor -> 36 baseline
#    accuracy failures. We rebuild the base-2 result from the natural-base
#    primitives: 2**(-|d|) = exp2(-|d| * ln2) and log2(z) = log2(z) * (1/ln2).

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def logaddexp2_func(x, y):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    # exp2(-|d|*ln2) == e**(-|d|*ln2) == 2**(-|d|); log2(z)*inv_ln2 == real log2.
    # Literals inlined (Triton @jit cannot read module-level globals).
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta) * 0.6931471805599453)) * (
        1.4426950408889634
    )
    # `delta` is NaN when x and y are equal infinities (inf - inf); result is m,
    # e.g. logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf.
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    return tl.where(is_nan, float("nan"), res)


def logaddexp2(self, other):
    logger.debug("GEMS_KUNLUNXIN LOGADDEXP2")
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN LOGADDEXP2_OUT")
    logaddexp2_func(self, other, out0=out)
    return out
