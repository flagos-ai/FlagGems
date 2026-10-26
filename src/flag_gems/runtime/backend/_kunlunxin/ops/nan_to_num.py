import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    # isCloseVectorization must stay False for the min/max sat-mul body below:
    # with it enabled the min/max ops measure ~27x slower than without (A/B
    # measured, 16.7M fp32: 11432us vs 135us). (The previous bit-compare +
    # tl.where body happened to profit from vectorization; the sat-mul body
    # does not, so this knob is body-specific.)
    isCloseVectorization=False,
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
    # Classify x into {finite, +inf, -inf, NaN} with pure saturated FP
    # arithmetic (min/max/mul only). The previous bit-compare + tl.where body
    # lowers to vcmp/vselect sequences that measure ~2.6x slower than the
    # pure-fp classify on this backend (same class of slow path as hardshrink /
    # not_equal). Each of m_pos / m_neg / m_fin is exactly 0.0 or 1.0:
    #   m_pos = 1 iff x == +inf;  m_neg = 1 iff x == -inf;
    #   m_fin = 1 iff x is finite (incl. +/-0.0, subnormals, +/-f32 max).
    # min/max here are non-NaN-propagating (maxnum/minnum), so the NaN case
    # falls out as m_fin == m_pos == m_neg == 0 -> m_nan == 1.
    # NOTE: all constants below are f32-inline literals; module-level
    # tl.constexpr Python floats are f64 on this backend and would promote the
    # whole body to the software-emulated f64 (a >10x regression).
    M = 3.4028234663852886e38  # exact f32 max finite
    m_pos = tl.minimum(tl.maximum((x - M) * 1e30 * 1e30, 0.0), 1.0)
    m_neg = tl.minimum(tl.maximum((0.0 - x - M) * 1e30 * 1e30, 0.0), 1.0)
    # (x - x) is exactly +0.0 for finite x and NaN for +/-inf/NaN input.
    m_fin = tl.minimum(tl.maximum(1.0 + (x - x) * 1e30 * 1e30, 0.0), 1.0)
    m_nan = 1.0 - (m_fin + m_pos + m_neg)
    # Blend with a nested select on the raw input -- never x in fp arithmetic:
    # this backend flushes subnormal f32/bf16 inputs to +0.0 in the fp add
    # (FTZ; verified: x + 0.0 == 0.0 for subnormal x), so the multiplied-then-
    # summed blend loses subnormal finite values. The masks are exactly 0/1
    # and mutually exclusive, so the select chain is a straight 4-way pick.
    return tl.where(
        m_fin > 0.5,
        x,
        tl.where(
            m_nan > 0.5,
            nan,
            tl.where(m_pos > 0.5, posinf, tl.where(m_neg > 0.5, neginf, 0.0)),
        ),
    )


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
    return nan_to_num_func(A, nan, posinf, neginf, out0=A)
