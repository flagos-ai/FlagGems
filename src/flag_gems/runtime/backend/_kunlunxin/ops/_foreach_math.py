# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XPU foreach overrides for the unary math ops the generic executor gets wrong.

``flag_gems.ops._foreach_unary`` reuses the *generic* per-operator scalar
kernels for the whole family.  Those kernels are built on
``tl_extra_shim``/``tl`` intrinsics, and on this backend several of them are
mis-lowered or inaccurate:

* ``acos``/``asin``: the XPU libdevice externs carry ~3e-3 absolute error
  (probed on [0.1, 0.9]); the vendor ``acos.py``/``asin.py`` polynomial bodies
  are the backend's accurate replacements (~2e-5).
* ``log2``: the ``tl.log2`` intrinsic drops the conversion factor on XPU, so
  the generic kernel computes ``ln`` (see ``log2.py`` for the probe); the vendor
  contract computes ``tl.log(x) * 1/ln(2)``.
* ``lgamma``: the vendor Lanczos implementation is the accurate one; the generic
  libdevice extern is what the single-tensor override exists to avoid.

The single-tensor overrides do not help ``_foreach_<op>``: the executor calls
the scalar ``fn`` directly, so the vendor math is bypassed.  These wrappers feed
the vendor scalar bodies into the shared executor instead, which keeps the
single-launch fast path (and the same dtype policy / complex fallback).

Everything else -- complex inputs, mixed dtypes/shapes, integral promotion --
is still handled by ``foreach_unary`` exactly as before.
"""

import logging

import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import UNARY_OPS, _scalar_fn
from flag_gems.utils.foreach import foreach_unary, real_dtype_of

from .acos import _acos_body
from .asin import _asin_body
from .lgamma import lgamma_func
from .log2 import log2_func

logger = logging.getLogger(__name__)

# The vendor bodies compute in fp32; the executor hands them the *output* dtype
# (fp16/bf16 stay narrow for a dtype-preserving op), so upcast here and cast the
# result back.  That mirrors what acos_kernel/asin_kernel do around the body.
_LGAMMA_FAST = _scalar_fn(lgamma_func)
_LOG2_FAST = _scalar_fn(log2_func)


@triton.jit
def _acos_fast(x):
    return _acos_body(x.to(tl.float32)).to(x.dtype)


@triton.jit
def _asin_fast(x):
    return _asin_body(x.to(tl.float32)).to(x.dtype)


def _apply_vendor(name, fn, self, inplace):
    """Run the shared executor with a vendor scalar function."""
    op = UNARY_OPS[name]
    if inplace:
        for t in self:
            if t.dtype.is_complex and op.out_dtype_fn is real_dtype_of:
                raise RuntimeError(
                    f"In-place {name} is not supported for complex tensors."
                )
        foreach_unary(
            self,
            fn,
            complex_fn=op.complex_fn,
            out_dtype_fn=op.out_dtype_fn,
            inplace=True,
            allowed_dtypes=op.allowed,
        )
        return None
    return foreach_unary(
        self,
        fn,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        allowed_dtypes=op.allowed,
    )


def _foreach_acos(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ACOS")
    return _apply_vendor("acos", _acos_fast, self, inplace=False)


def _foreach_acos_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ACOS_")
    return _apply_vendor("acos", _acos_fast, self, inplace=True)


def _foreach_asin(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ASIN")
    return _apply_vendor("asin", _asin_fast, self, inplace=False)


def _foreach_asin_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ASIN_")
    return _apply_vendor("asin", _asin_fast, self, inplace=True)


def _foreach_log2(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LOG2")
    return _apply_vendor("log2", _LOG2_FAST, self, inplace=False)


def _foreach_log2_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LOG2_")
    return _apply_vendor("log2", _LOG2_FAST, self, inplace=True)


def _foreach_lgamma(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LGAMMA")
    return _apply_vendor("lgamma", _LGAMMA_FAST, self, inplace=False)


def _foreach_lgamma_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LGAMMA_")
    return _apply_vendor("lgamma", _LGAMMA_FAST, self, inplace=True)
