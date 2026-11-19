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

"""GCU300 fallbacks for the unary ``_foreach_*`` family.

The shared executor in :mod:`flag_gems.utils.foreach` aborts the process on
GCU300 (driver ``Sip exception``, rc=137) for every unary ``_foreach_*`` call.
These wrappers keep the ATen contract -- dtype promotion, in-place refusal,
overlap refusal and the empty-list error -- and apply the existing single-tensor
kernel to one tensor at a time, so no symbol from the shared executor is
referenced.

``_foreach_asin`` / ``_foreach_asin_`` / ``_foreach_erfc`` / ``_foreach_erfc_``
are deliberately absent: the single-tensor ``asin`` / ``erfc`` kernels abort on
GCU300 (rc=134), so they are tracked as a follow-up rather than shipped here.
"""

import logging

import flag_gems
from flag_gems.ops._foreach_unary import UNARY_OPS
from flag_gems.utils.foreach import (
    check_dtype_supported,
    check_tensor_list,
    has_internal_overlap,
)

logger = logging.getLogger(__name__)


def _log(name: str, inplace: bool) -> None:
    logger.debug("GEMS _FOREACH_%s%s", name.upper(), "_" if inplace else "")


def _apply(name: str, self, inplace: bool):
    tensors = check_tensor_list(self)
    op = UNARY_OPS[name]
    results = []
    for t in tensors:
        if t.dtype.is_complex:
            # gcu300 has no complex datapath; the generic executor's complex
            # kernels cannot run here and no complex tensor can be created by
            # this backend anyway.
            raise NotImplementedError(
                f"foreach executor has no complex kernel for {t.dtype}"
            )
        if op.allowed is not None and t.dtype not in op.allowed:
            check_dtype_supported(t.dtype, op.allowed)
        out_dtype = op.out_dtype_fn(t.dtype)
        if inplace:
            if out_dtype != t.dtype:
                # Same refusal as ATen: an in-place op may not silently narrow
                # its promoted result back into the input's dtype.
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(t.dtype).replace('torch.', '').capitalize()}"
                )
            if has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            # Compute out of place and copy back: several single-tensor
            # in-place kernels (notably ``floor_`` on integral tensors) crash
            # on GCU300, while the functional kernel is healthy and ``copy_``
            # still lands in the original storage, including gappy views.
            work = t if t.is_contiguous() else t.contiguous()
            t.copy_(getattr(flag_gems, name)(work))
            results.append(t)
        else:
            # Integral inputs of a promoting operator are cast before the
            # single-tensor kernel, matching foreach's int_to_float policy.
            # Gappy views are staged: several single-tensor kernels (``round``
            # among them) only accept contiguous input.
            work = t if t.is_contiguous() else t.contiguous()
            work = work if work.dtype == out_dtype else work.to(out_dtype)
            results.append(getattr(flag_gems, name)(work))
    return None if inplace else results


def _foreach_abs(self):
    _log("abs", False)
    return _apply("abs", self, False)


def _foreach_abs_(self):
    _log("abs", True)
    return _apply("abs", self, True)


def _foreach_acos_(self):
    _log("acos", True)
    return _apply("acos", self, True)


def _foreach_atan_(self):
    _log("atan", True)
    return _apply("atan", self, True)


def _foreach_cos_(self):
    _log("cos", True)
    return _apply("cos", self, True)


def _foreach_cosh_(self):
    _log("cosh", True)
    return _apply("cosh", self, True)


def _foreach_erf_(self):
    _log("erf", True)
    return _apply("erf", self, True)


def _foreach_exp_(self):
    _log("exp", True)
    return _apply("exp", self, True)


def _foreach_expm1_(self):
    _log("expm1", True)
    return _apply("expm1", self, True)


def _foreach_floor_(self):
    _log("floor", True)
    return _apply("floor", self, True)


def _foreach_log10_(self):
    _log("log10", True)
    return _apply("log10", self, True)


def _foreach_log1p_(self):
    _log("log1p", True)
    return _apply("log1p", self, True)


def _foreach_log2_(self):
    _log("log2", True)
    return _apply("log2", self, True)


def _foreach_neg_(self):
    _log("neg", True)
    return _apply("neg", self, True)


def _foreach_reciprocal_(self):
    _log("reciprocal", True)
    return _apply("reciprocal", self, True)


def _foreach_round_(self):
    _log("round", True)
    return _apply("round", self, True)


def _foreach_rsqrt_(self):
    _log("rsqrt", True)
    return _apply("rsqrt", self, True)


def _foreach_sigmoid_(self):
    _log("sigmoid", True)
    return _apply("sigmoid", self, True)


def _foreach_sign(self):
    _log("sign", False)
    return _apply("sign", self, False)


def _foreach_sign_(self):
    _log("sign", True)
    return _apply("sign", self, True)


def _foreach_sin_(self):
    _log("sin", True)
    return _apply("sin", self, True)


def _foreach_sinh_(self):
    _log("sinh", True)
    return _apply("sinh", self, True)


def _foreach_sqrt_(self):
    _log("sqrt", True)
    return _apply("sqrt", self, True)


def _foreach_tan_(self):
    _log("tan", True)
    return _apply("tan", self, True)


def _foreach_tanh_(self):
    _log("tanh", True)
    return _apply("tanh", self, True)


def _foreach_trunc(self):
    _log("trunc", False)
    return _apply("trunc", self, False)


def _foreach_trunc_(self):
    _log("trunc", True)
    return _apply("trunc", self, True)


__all__ = [
    "_foreach_abs",
    "_foreach_abs_",
    "_foreach_acos_",
    "_foreach_atan_",
    "_foreach_cos_",
    "_foreach_cosh_",
    "_foreach_erf_",
    "_foreach_exp_",
    "_foreach_expm1_",
    "_foreach_floor_",
    "_foreach_log10_",
    "_foreach_log1p_",
    "_foreach_log2_",
    "_foreach_neg_",
    "_foreach_reciprocal_",
    "_foreach_round_",
    "_foreach_rsqrt_",
    "_foreach_sigmoid_",
    "_foreach_sign",
    "_foreach_sign_",
    "_foreach_sin_",
    "_foreach_sinh_",
    "_foreach_sqrt_",
    "_foreach_tan_",
    "_foreach_tanh_",
    "_foreach_trunc",
    "_foreach_trunc_",
]
