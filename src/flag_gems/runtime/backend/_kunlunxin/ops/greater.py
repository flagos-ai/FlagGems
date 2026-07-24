# Kunlunxin (XPU) override of greater / greater_out / greater_scalar /
# greater_scalar_out.
#
# `greater.Tensor` is functionally identical to `gt.Tensor`, and kunlunxin
# already ships a tuned override for gt (`_kunlunxin/ops/gt.py`). But `greater`
# was NOT overridden, so it fell back to the generic bare `pointwise_dynamic`
# (no CodeGenConfig) -> discrete access on XPU -> catastrophic latency
# (60-1000 ms for large shapes, gems speedup ~0.001 in
# `harness/perf_ir_2/greater.log`).
#
# Fix: reuse the exact gt recipe -- same tuned CodeGenConfig
# (unroll_num=8, kunlunAutoGrid=True, prefer_1d_tile=True) plus the
# TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST launch env vars for the tensor
# path. Kernel body / algorithm unchanged (zero correctness risk).
import logging
import os

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
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


# Scalar (tensor-vs-scalar) compare path. Same bandwidth-bound 1D-tile recipe
# as config_, but with unroll_num=16 + buffer_size_limit=8192. On XPU the scalar
# greater kernel is pure memory-bound (~385 GB/s at unroll_num=8); a fresh-compile
# config sweep on [1024,1024,1024] showed unroll_num=16 + buffer_size_limit=8192
# is the sweet spot -> fp16 7.85->6.84ms, fp32 7.31->6.00ms (~13-18% faster),
# while unroll_num=32 and larger buffer_size_limit regress or plateau. Pure
# codegen-param change: kernel body / algorithm / numerics unchanged.
# NOTE: the fusion env vars used by the tensor path (TRITONXPU_COMPARE_FUSION /
# TRITONXPU_FP16_FAST) are deliberately NOT used here -- a fresh-compile sweep
# proved they give zero latency benefit on the scalar kernel AND TRITONXPU_FP16_FAST
# triggers an `out of resource: uni_sram` compile failure for fp16.
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
    # NOTE: unlike the tensor path, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST. For tensor-vs-scalar
    # compare these fusion env vars make the compiler emit an fp16 compare that
    # trips `arith.cmpf requires all operands to have the same type` and blows the
    # uni_sram budget -> `out of resource: uni_sram` compile failure (fp16). The
    # sibling gt_scalar deliberately omits them for the same reason.
    res = greater_func_scalar(A, B)
    return res


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR_OUT")
    # See greater_scalar: no fusion env vars on the scalar path (fp16 compile).
    if out is None:
        res = greater_func_scalar(A, B)
    else:
        greater_func_scalar(A, B, out0=out)
        res = out
    return res
