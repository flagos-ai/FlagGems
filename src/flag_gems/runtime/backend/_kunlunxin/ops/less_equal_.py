# Kunlunxin (XPU) override of less_equal_ / less_equal_scalar_ (in-place).
#
# `less_equal_.Tensor` is the in-place variant of `less_equal.Tensor`, and
# kunlunxin already ships a tuned override for the out-of-place `less_equal`
# (`_kunlunxin/ops/less_equal.py`, itself reusing the `le.py` recipe). But
# `less_equal_` was NOT overridden, so it fell back to the generic bare
# `pointwise_dynamic` (no CodeGenConfig) -> discrete access on XPU ->
# catastrophic latency (4096x4096 fp32: gems 50.8ms vs torch 0.128ms,
# speedup 0.003; dtype-balanced 0.082, 2026-09-03 baseline).
#
# Fix: reuse the tuned CodeGenConfig (block=1024, unroll_num=8,
# kunlunAutoGrid=True, prefer_1d_tile=True) plus the
# TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST launch env vars for the tensor
# path; in-place write-back via `out0=A`. Kernel body uses an arithmetic-only
# formulation of `x <= y` (see less_equal_func) because every TritonXPU
# select/compare path is capped at ~60 Gelem/s by the vselect mask assembly
# (dtype-balanced 0.082 -> 0.509 -> 0.949, 2026-09-03).
import logging
import os

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


config_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def less_equal_func(x, y):
    # x <= y <=> !(y - x < 0). 算术化实现，绕开 TritonXPU 的 vselect 掩码组装：
    # VSelectOpConversion 把谓词掩码逐位组装（每 32 元素 ~16 条 select+or），
    # 任何 select/compare 路径的带宽被锁死在 ~60 Gelem/s（fp16 where 365GB/s、
    # fp32 710GB/s，2026-09-03 实测）；而 min/max/clip 是纯 VALU 指令，
    # 接近 copy 全速（fp16 1588GB/s）。本公式全部用 fast 类指令：
    #   d   = y - x
    #   neg = min(d, 0)     # d<0 时为 d，否则 0
    #   mag = max(-neg, 0)  # d<0 时为 -d，否则 0
    #   r   = 1 - min(mag * 1e38, 1)
    # d<0 -> mag>0 -> mag*1e38 溢出为 inf -> min 取 1 -> r=0；d>=0 -> r=1。
    # ±0 正确（d=±0 -> mag=0 -> r=1）。NaN 行为与 where 版一致（对非 0/1 输入）。
    # 乘数 1e38 必须保持 f32（f16 下 0*inf=NaN 会破坏 d=0 分支）。
    d = y - x
    neg = tl.minimum(d, 0.0)
    mag = tl.maximum(-neg, 0.0)
    return 1.0 - tl.minimum(mag * 1e38, 1.0)


def less_equal_(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = less_equal_func(A, B, out0=A)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return A


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def less_equal_func_scalar(x, y):
    # 与 tensor 路径相同的算术化实现（见 less_equal_func 注释）。
    d = y - x
    neg = tl.minimum(d, 0.0)
    mag = tl.maximum(-neg, 0.0)
    return 1.0 - tl.minimum(mag * 1e38, 1.0)


def less_equal_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_SCALAR_")
    # NOTE: unlike the tensor path, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST. For tensor-vs-scalar
    # compare these fusion env vars make the compiler emit an fp16 compare that
    # trips `arith.cmpf requires all operands to have the same type` and blows
    # the uni_sram budget -> `out of resource: uni_sram` compile failure (fp16).
    # The sibling le_scalar / gt_scalar / less_equal_scalar deliberately omit
    # them for the same reason.
    res = less_equal_func_scalar(A, B, out0=A)
    return A
