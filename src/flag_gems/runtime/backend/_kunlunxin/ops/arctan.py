import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic
from .atan import _launch as _poly_launch

_atan = tl_extra_shim.atan
logger = logging.getLogger(__name__)

# arctan is an alias of atan (out = atan(x)). arctan/arctan_ dispatch to the
# `aten::arctan`/`aten::arctan_` schemas (torch.arctan is NOT the same
# callable as torch.atan in torch 2.9.0) and are overridden here in
# _kunlunxin/ops.
#
# Baseline (XPU 2, 2026-08-19, official matrix = UnaryPointwiseBenchmark
# 12 shapes x {fp16,fp32,bf16} = 36 rows/op, recorded via
# benchmark/test_atan.py proxy + identical arctan probe): the previous impl
# was a bare pointwise_dynamic kernel calling the xpu::atanf extern
# elementwise shim (~40ns/lane scalar llvm.call), dtype-equal Gems Speedup
# 0.6966x (arctan) / 0.6914x (arctan_).
#
# Fix (all local to this file; atan.py stays untouched):
#  * fp16/fp32 at all sizes -> the atan.py poly kernel (single-select
#    algebraic form: odd deg-11 minimax poly carrying the sign, x*|1/x| = +-1
#    pip term; unmasked BLOCK=8192/2048 fast paths + masked fallback). This
#    is the exact kernel validated by atan (fp32 max abs err 1.56e-5 / fp16
#    4.88e-4 / bf16 3.89e-3, all in-official-tolerance on the same
#    POINTWISE_SHAPES x FLOAT_DTYPES functional matrix).
#  * bf16 with numel >= 2^20 keeps the ORIGINAL tuned-extern kernel
#    (config_ below): measured probe poly-bf16 16M elems = 791.9us vs tuned
#    extern 700.3-700.7us (baseline), i.e. a 1.6%-13% regression on the big
#    bf16 shapes, while fp16/fp32 poly gains ~9-15% there. Small bf16
#    (< 2^20) goes to poly (-31%..-77% measured).
#  * in-place uses the poly kernel directly (A.is_contiguous fast path);
#    the 2D (4096, 65536]-numel extern window is NOT used because the bare
#    extern in-place path measures 99.8us at (1024,16) vs 7.2us poly.

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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def arctan_func(x):
    return _atan(x.to(tl.float32))


_BF16_EXTERN_MIN = 2**24  # 16M; see docstring for the measured crossover


def _use_tuned_extern(A, n):
    return A.dtype == torch.bfloat16 and n >= _BF16_EXTERN_MIN


def arctan(A):
    logger.debug("GEMS_KUNLUNXIN ARCTAN")
    n = A.numel()
    if _use_tuned_extern(A, n):
        return arctan_func(A)
    out = torch.empty_like(A)
    _poly_launch(A.contiguous(), out)
    return out


def arctan_(A):
    logger.debug("GEMS_KUNLUNXIN ARCTAN_")
    n = A.numel()
    if _use_tuned_extern(A, n):
        arctan_func(A, out0=A)
        return A
    if A.is_contiguous():
        _poly_launch(A, A)
        return A
    tmp = A.contiguous()
    _poly_launch(tmp, tmp)
    A.copy_(tmp)
    return A