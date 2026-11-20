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

"""GCU300 fallbacks for the binary ``_foreach_*`` family.

The shared executor in :mod:`flag_gems.utils.foreach` either aborts the process
(unary/reduction path) or fails to link (the ``tops_extui_v1_ui16_to_v4_ui64``
relocation) on GCU300.  These wrappers keep the ATen dtype/error contract and
apply the existing single-tensor kernel to one tensor at a time.  Non-contiguous
operands are staged to dense form before the call, mirroring the generic
executor's ``_foreach_binary`` pairing semantics; in-place writes are copied
back to the original view.

The file covers the 44 overloads in the Enflame acceptance list.  Sibling
overloads that are not in that list (e.g. ``add.List``, ``add.Tensor``) keep the
generic implementation and are out of scope here.
"""

import logging

import torch

import flag_gems
from flag_gems.ops._foreach_binary import _POW_BOOL_ERROR, BINARY_OPS
from flag_gems.utils.foreach import (
    _check_same_length,
    _promote_dtypes,
    _promote_with_scalar,
    check_dtype_supported,
    check_tensor_list,
    has_internal_overlap,
)

logger = logging.getLogger(__name__)

# core name -> single-tensor operator; ``div`` is exposed as ``true_divide``
# and ``pow`` / the ternary operators have dedicated paths below.
_SINGLE = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "true_divide",
    "clamp_max": "clamp_max",
    "clamp_min": "clamp_min",
    "maximum": "maximum",
    "minimum": "minimum",
}


def _log(core: str, inplace: bool) -> None:
    logger.debug("GEMS _FOREACH_%s%s", core.upper(), "_" if inplace else "")


def _check_allowed(op, dtype: torch.dtype, scalar: bool = False) -> None:
    allowed = (
        op.allowed_scalar if scalar and op.allowed_scalar is not None else op.allowed
    )
    if allowed is not None and dtype not in allowed:
        check_dtype_supported(dtype, allowed, op.dtype_error, op.dtype_error_type)


def _check_inplace(dtype: torch.dtype, out_dtype: torch.dtype) -> None:
    if out_dtype != dtype:
        raise RuntimeError(
            f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
            " can't be cast to the desired output type "
            f"{str(dtype).replace('torch.', '').capitalize()}"
        )


def _stage(t):
    if not isinstance(t, torch.Tensor):
        return t
    return t if t.is_contiguous() else t.contiguous()


def _apply_one(core: str, a: torch.Tensor, b, inplace: bool, scalar: bool = False):
    """Apply one binary pair, staging non-contiguous operands."""
    op = BINARY_OPS[core]
    _check_allowed(op, a.dtype, scalar=scalar)
    if isinstance(b, torch.Tensor):
        out_dtype = _promote_dtypes(a, b)
    else:
        out_dtype = _promote_with_scalar(a.dtype, b)
    if inplace:
        _check_inplace(a.dtype, out_dtype)
        if has_internal_overlap(a):
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )

    # ``maximum`` / ``minimum`` are tensor-tensor kernels; materialise the
    # scalar as a 0-dim tensor in the promoted dtype so the single-tensor op
    # performs exactly the promotion ``_promote_with_scalar`` just computed.
    if core in ("maximum", "minimum") and not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b, dtype=out_dtype, device=a.device)

    # ``add`` on bool is logical or in ATen; arithmetic on int1 would be XOR.
    if core == "add" and a.dtype == torch.bool:
        res = a | b
        if inplace:
            a.copy_(res)
            return None
        return res

    name = _SINGLE[core]
    if inplace:
        single_in = getattr(flag_gems, name + "_", None)
        if single_in is None:
            # ``maximum_`` / ``minimum_`` have no single-tensor in-place
            # kernel; compute out of place and copy back.
            a.copy_(getattr(flag_gems, name)(_stage(a), _stage(b)))
        elif a.is_contiguous():
            single_in(a, _stage(b))
        else:
            work = a.contiguous()
            single_in(work, _stage(b))
            a.copy_(work)
        return None
    return getattr(flag_gems, name)(_stage(a), _stage(b))


def _apply_scalar(core: str, self, scalar, inplace: bool):
    _log(core, inplace)
    tensors = check_tensor_list(self)
    results = []
    for t in tensors:
        res = _apply_one(core, t, scalar, inplace, scalar=True)
        results.append(t if inplace else res)
    return None if inplace else results


def _apply_scalar_list(core: str, self, scalars, inplace: bool):
    _log(core, inplace)
    tensors = check_tensor_list(self)
    scalars = list(scalars)
    _check_same_length(tensors, scalars, "scalars")
    results = []
    for t, s in zip(tensors, scalars):
        res = _apply_one(core, t, s, inplace, scalar=True)
        results.append(t if inplace else res)
    return None if inplace else results


def _apply_tensor(core: str, self, other, inplace: bool):
    _log(core, inplace)
    if not isinstance(other, torch.Tensor):
        raise TypeError("argument 'other' must be a Tensor")
    tensors = check_tensor_list(self)
    results = []
    for t in tensors:
        operand = other if other.numel() == 1 else other.expand_as(t)
        res = _apply_one(core, t, operand, inplace)
        results.append(t if inplace else res)
    return None if inplace else results


def _value_at(value, i: int):
    if isinstance(value, (list, tuple)):
        return value[i]
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value[i].item()
    return value


def _apply_ternary(core: str, self, tensor1, tensor2, value, inplace: bool):
    _log(core, inplace)
    op = BINARY_OPS[core]
    tensors = check_tensor_list(self)
    t1 = list(tensor1)
    t2 = list(tensor2)
    _check_same_length(tensors, t1, "tensor1")
    _check_same_length(tensors, t2, "tensor2")
    single = getattr(flag_gems, core + ("_" if inplace else ""))
    results = []
    for i, (a, b, c) in enumerate(zip(tensors, t1, t2)):
        _check_allowed(op, a.dtype)
        out_dtype = _promote_dtypes(a, b, c)
        if inplace:
            _check_inplace(a.dtype, out_dtype)
            if has_internal_overlap(a):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
        v = _value_at(value, i)
        if inplace:
            if a.is_contiguous():
                single(a, _stage(b), _stage(c), value=v)
            else:
                work = a.contiguous()
                single(work, _stage(b), _stage(c), value=v)
                a.copy_(work)
            results.append(a)
        else:
            results.append(single(_stage(a), _stage(b), _stage(c), value=v))
    return None if inplace else results


# --- target wrappers -------------------------------------------------------


def _foreach_add_Scalar(self, scalar):
    return _apply_scalar("add", self, scalar, False)


def _foreach_add__Scalar(self, scalar):
    return _apply_scalar("add", self, scalar, True)


def _foreach_add_ScalarList(self, scalars):
    return _apply_scalar_list("add", self, scalars, False)


def _foreach_add__ScalarList(self, scalars):
    return _apply_scalar_list("add", self, scalars, True)


def _foreach_sub_Scalar(self, scalar):
    return _apply_scalar("sub", self, scalar, False)


def _foreach_sub__Scalar(self, scalar):
    return _apply_scalar("sub", self, scalar, True)


def _foreach_sub_ScalarList(self, scalars):
    return _apply_scalar_list("sub", self, scalars, False)


def _foreach_sub__ScalarList(self, scalars):
    return _apply_scalar_list("sub", self, scalars, True)


def _foreach_mul_Scalar(self, scalar):
    return _apply_scalar("mul", self, scalar, False)


def _foreach_mul__Scalar(self, scalar):
    return _apply_scalar("mul", self, scalar, True)


def _foreach_mul_ScalarList(self, scalars):
    return _apply_scalar_list("mul", self, scalars, False)


def _foreach_mul__ScalarList(self, scalars):
    return _apply_scalar_list("mul", self, scalars, True)


def _foreach_mul__Tensor(self, other):
    return _apply_tensor("mul", self, other, True)


def _foreach_div_Scalar(self, scalar):
    return _apply_scalar("div", self, scalar, False)


def _foreach_div__Scalar(self, scalar):
    return _apply_scalar("div", self, scalar, True)


def _foreach_div_ScalarList(self, scalars):
    return _apply_scalar_list("div", self, scalars, False)


def _foreach_div__ScalarList(self, scalars):
    return _apply_scalar_list("div", self, scalars, True)


def _foreach_div__Tensor(self, other):
    return _apply_tensor("div", self, other, True)


def _foreach_clamp_max_Scalar(self, scalar):
    return _apply_scalar("clamp_max", self, scalar, False)


def _foreach_clamp_max__Scalar(self, scalar):
    return _apply_scalar("clamp_max", self, scalar, True)


def _foreach_clamp_max_ScalarList(self, scalars):
    return _apply_scalar_list("clamp_max", self, scalars, False)


def _foreach_clamp_max__ScalarList(self, scalars):
    return _apply_scalar_list("clamp_max", self, scalars, True)


def _foreach_clamp_min_Scalar(self, scalar):
    return _apply_scalar("clamp_min", self, scalar, False)


def _foreach_clamp_min__Scalar(self, scalar):
    return _apply_scalar("clamp_min", self, scalar, True)


def _foreach_clamp_min_ScalarList(self, scalars):
    return _apply_scalar_list("clamp_min", self, scalars, False)


def _foreach_maximum_Scalar(self, scalar):
    return _apply_scalar("maximum", self, scalar, False)


def _foreach_maximum__Scalar(self, scalar):
    return _apply_scalar("maximum", self, scalar, True)


def _foreach_maximum_ScalarList(self, scalars):
    return _apply_scalar_list("maximum", self, scalars, False)


def _foreach_maximum__ScalarList(self, scalars):
    return _apply_scalar_list("maximum", self, scalars, True)


def _foreach_minimum_Scalar(self, scalar):
    return _apply_scalar("minimum", self, scalar, False)


def _foreach_minimum__Scalar(self, scalar):
    return _apply_scalar("minimum", self, scalar, True)


def _foreach_minimum_ScalarList(self, scalars):
    return _apply_scalar_list("minimum", self, scalars, False)


def _foreach_minimum__ScalarList(self, scalars):
    return _apply_scalar_list("minimum", self, scalars, True)


def _foreach_addcmul__Scalar(self, tensor1, tensor2, value=1):
    return _apply_ternary("addcmul", self, tensor1, tensor2, value, True)


def _foreach_addcmul__ScalarList(self, tensor1, tensor2, value=1):
    return _apply_ternary("addcmul", self, tensor1, tensor2, value, True)


def _foreach_addcmul__Tensor(self, tensor1, tensor2, value=1):
    return _apply_ternary("addcmul", self, tensor1, tensor2, value, True)


def _foreach_addcdiv__Scalar(self, tensor1, tensor2, value=1):
    return _apply_ternary("addcdiv", self, tensor1, tensor2, value, True)


def _foreach_addcdiv__ScalarList(self, tensor1, tensor2, value=1):
    return _apply_ternary("addcdiv", self, tensor1, tensor2, value, True)


def _foreach_addcdiv__Tensor(self, tensor1, tensor2, value=1):
    return _apply_ternary("addcdiv", self, tensor1, tensor2, value, True)


def _foreach_pow_List(self, exponent):
    _log("pow", False)
    op = BINARY_OPS["pow"]
    tensors = check_tensor_list(self)
    exps = list(exponent)
    _check_same_length(tensors, exps, "exponent")
    results = []
    for a, b in zip(tensors, exps):
        _check_allowed(op, a.dtype)
        results.append(flag_gems.pow_tensor_tensor(_stage(a), _stage(b)))
    return results


def _foreach_pow__List(self, exponent):
    _log("pow", True)
    op = BINARY_OPS["pow"]
    tensors = check_tensor_list(self)
    exps = list(exponent)
    _check_same_length(tensors, exps, "exponent")
    for a, b in zip(tensors, exps):
        _check_allowed(op, a.dtype)
        _check_inplace(a.dtype, _promote_dtypes(a, b))
        if has_internal_overlap(a):
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        if a.is_contiguous():
            flag_gems.pow_tensor_tensor_(a, _stage(b))
        else:
            work = a.contiguous()
            flag_gems.pow_tensor_tensor_(work, _stage(b))
            a.copy_(work)
    return None


def _foreach_pow__Scalar(self, exponent):
    _log("pow", True)
    op = BINARY_OPS["pow"]
    tensors = check_tensor_list(self)
    for a in tensors:
        _check_allowed(op, a.dtype, scalar=True)
        _check_inplace(a.dtype, _promote_with_scalar(a.dtype, exponent))
        if has_internal_overlap(a):
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        if a.is_contiguous():
            flag_gems.pow_tensor_scalar_(a, exponent)
        else:
            work = a.contiguous()
            flag_gems.pow_tensor_scalar_(work, exponent)
            a.copy_(work)
    return None


def _foreach_pow__ScalarList(self, exponents):
    _log("pow", True)
    op = BINARY_OPS["pow"]
    tensors = check_tensor_list(self)
    exps = list(exponents)
    _check_same_length(tensors, exps, "exponent")
    for a, e in zip(tensors, exps):
        _check_allowed(op, a.dtype, scalar=True)
        _check_inplace(a.dtype, _promote_with_scalar(a.dtype, e))
        if has_internal_overlap(a):
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        if a.is_contiguous():
            flag_gems.pow_tensor_scalar_(a, e)
        else:
            work = a.contiguous()
            flag_gems.pow_tensor_scalar_(work, e)
            a.copy_(work)
    return None


def _foreach_pow_ScalarAndTensor(self, exponent):
    _log("pow", False)
    exps = list(exponent)
    for t in exps:
        if t.dtype == torch.bool:
            raise NotImplementedError(_POW_BOOL_ERROR)
    return [flag_gems.pow_scalar(self, _stage(t)) for t in exps]


__all__ = [
    "_foreach_add_Scalar",
    "_foreach_add__Scalar",
    "_foreach_add_ScalarList",
    "_foreach_add__ScalarList",
    "_foreach_sub_Scalar",
    "_foreach_sub__Scalar",
    "_foreach_sub_ScalarList",
    "_foreach_sub__ScalarList",
    "_foreach_mul_Scalar",
    "_foreach_mul__Scalar",
    "_foreach_mul_ScalarList",
    "_foreach_mul__ScalarList",
    "_foreach_mul__Tensor",
    "_foreach_div_Scalar",
    "_foreach_div__Scalar",
    "_foreach_div_ScalarList",
    "_foreach_div__ScalarList",
    "_foreach_div__Tensor",
    "_foreach_clamp_max_Scalar",
    "_foreach_clamp_max__Scalar",
    "_foreach_clamp_max_ScalarList",
    "_foreach_clamp_max__ScalarList",
    "_foreach_clamp_min_Scalar",
    "_foreach_clamp_min__Scalar",
    "_foreach_clamp_min_ScalarList",
    "_foreach_maximum_Scalar",
    "_foreach_maximum__Scalar",
    "_foreach_maximum_ScalarList",
    "_foreach_maximum__ScalarList",
    "_foreach_minimum_Scalar",
    "_foreach_minimum__Scalar",
    "_foreach_minimum_ScalarList",
    "_foreach_minimum__ScalarList",
    "_foreach_addcmul__Scalar",
    "_foreach_addcmul__ScalarList",
    "_foreach_addcmul__Tensor",
    "_foreach_addcdiv__Scalar",
    "_foreach_addcdiv__ScalarList",
    "_foreach_addcdiv__Tensor",
    "_foreach_pow_List",
    "_foreach_pow__List",
    "_foreach_pow__Scalar",
    "_foreach_pow__ScalarList",
    "_foreach_pow_ScalarAndTensor",
]
