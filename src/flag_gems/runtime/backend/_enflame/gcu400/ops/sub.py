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

import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


# ---------------------------------------------------------------------------
# Complex support (parity with the generic sub, adapted for torch_gcu).
#
# flagtree's triton cannot take complex dtypes (canonicalize_dtype raises
# KeyError: 'complex64') and torch_gcu's topsaten has no complex sub/cast
# kernels (P21), so complex inputs must be decomposed into the float "pair"
# domain before any kernel launch.  The sibling enflame add op already ships
# that decomposition (view_as_real / pair-domain kernel / view_as_complex);
# since A - B * alpha == A + B * (-alpha) for real alpha, the complex path
# simply composes that FlagGems op with a negated alpha.  Real inputs keep
# the original sub_func kernels untouched.
#
# The add import must stay function-local: pointwise_dynamic's codegen
# sweeps every module-level ImportFrom of this file into the generated
# kernel modules, and a relative `.add` import cannot be resolved there
# (the backend is also loadable as a top-level `gcu400` package).
# ---------------------------------------------------------------------------


def _sub_complex(A, B, alpha):
    if isinstance(alpha, complex):
        raise NotImplementedError(
            "sub with complex alpha is not supported on the enflame backend"
        )
    from .add import add as _gems_add

    return _gems_add(A, B, alpha=-alpha)


def sub(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.is_complex() or B.is_complex():
            return _sub_complex(A, B, alpha)
        return sub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        if A.is_complex() or isinstance(B, complex):
            return _sub_complex(A, B, alpha)
        return sub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        if B.is_complex() or isinstance(A, complex):
            return _sub_complex(A, B, alpha)
        return sub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME SUB_")
    if isinstance(B, torch.Tensor):
        if A.is_complex() or B.is_complex():
            A.copy_(_sub_complex(A, B, alpha))
            return A
        return sub_func(A, B, alpha, out0=A)
    else:
        if A.is_complex() or isinstance(B, complex):
            A.copy_(_sub_complex(A, B, alpha))
            return A
        return sub_func_tensor_scalar(A, B, alpha, out0=A)
