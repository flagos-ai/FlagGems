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

"""Narrow compatibility shims for gaps in Kunlunxin's native ATen kernels."""

import warnings

import torch

_AD_INPLACE_OR_VIEW_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.ADInplaceOrView
)
_BACKEND_SELECT_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.BackendSelect)


def _redispatch_below_ad_inplace_or_view(op, keyset, *args, **kwargs):
    """Continue through native autograd and the currently active CUDA kernel."""
    return op.redispatch(keyset - _AD_INPLACE_OR_VIEW_KEYSET, *args, **kwargs)


def _is_klx_tensor(value):
    return isinstance(value, torch.Tensor) and value.device.type == "cuda"


def _is_complex_operand(value):
    return isinstance(value, complex) or (
        isinstance(value, torch.Tensor) and value.is_complex()
    )


def _is_fp64_compat_result(self, other):
    from ._fp64_compat import is_fp64_cuda_tensor

    return (
        is_fp64_cuda_tensor(self)
        and not _is_complex_operand(other)
        and torch.result_type(self, other) == torch.float64
    )


def _complex_scalar_tensor(anchor, value):
    result_dtype = torch.result_type(anchor, value)
    component_dtype = {
        torch.complex32: torch.float16,
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }[result_dtype]
    real = torch.scalar_tensor(value.real, dtype=component_dtype, device=anchor.device)
    imag = torch.scalar_tensor(value.imag, dtype=component_dtype, device=anchor.device)
    return torch.view_as_complex(torch.stack((real, imag)))


def _add_tensor_compat(keyset, self, other, *, alpha=1):
    self_is_tensor = isinstance(self, torch.Tensor)
    other_is_tensor = isinstance(other, torch.Tensor)
    klx_tensor = (
        self if _is_klx_tensor(self) else other if _is_klx_tensor(other) else None
    )

    if klx_tensor is not None and (
        not (self_is_tensor and other_is_tensor)
        or _is_complex_operand(self)
        or _is_complex_operand(other)
    ):
        if isinstance(self, complex):
            self = _complex_scalar_tensor(klx_tensor, self)
        if isinstance(other, complex):
            other = _complex_scalar_tensor(klx_tensor, other)
        # Import lazily to avoid a cycle while the KLX ops package is loading.
        from .add import add

        return add(self, other, alpha=alpha)

    if self_is_tensor and not other_is_tensor:
        return _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.add.Scalar, keyset, self, other, alpha=alpha
        )
    if not self_is_tensor and other_is_tensor:
        scaled_other = _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.mul.Scalar, keyset, other, alpha
        )
        return _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.add.Scalar, keyset, scaled_other, self
        )
    if not self_is_tensor and not other_is_tensor:
        self = torch.scalar_tensor(self)
        return _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.add.Scalar, keyset, self, other, alpha=alpha
        )

    return _redispatch_below_ad_inplace_or_view(
        torch.ops.aten.add.Tensor, keyset, self, other, alpha=alpha
    )


def _sub_tensor_compat(keyset, self, other, *, alpha=1):
    self_is_tensor = isinstance(self, torch.Tensor)
    other_is_tensor = isinstance(other, torch.Tensor)
    self_is_bool = (self_is_tensor and self.dtype == torch.bool) or isinstance(
        self, bool
    )
    other_is_bool = (other_is_tensor and other.dtype == torch.bool) or isinstance(
        other, bool
    )
    if self_is_bool or other_is_bool:
        if self_is_bool and other_is_bool:
            raise RuntimeError(
                "Subtraction, the `-` operator, with two bool tensors is not "
                "supported. Use the `^` or `logical_xor()` operator instead."
            )
        raise RuntimeError(
            "Subtraction, the `-` operator, with a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or "
            "`logical_not()` operator instead."
        )

    if _is_fp64_compat_result(self, other) and not _is_complex_operand(alpha):
        from ._fp64_compat import sub_fp64

        return sub_fp64(self, other, alpha=alpha)

    klx_tensor = (
        self if _is_klx_tensor(self) else other if _is_klx_tensor(other) else None
    )
    if klx_tensor is not None and (
        not (self_is_tensor and other_is_tensor)
        or _is_complex_operand(self)
        or _is_complex_operand(other)
    ):
        if isinstance(self, complex):
            self = _complex_scalar_tensor(klx_tensor, self)
        if isinstance(other, complex):
            other = _complex_scalar_tensor(klx_tensor, other)
        # Import lazily to avoid a cycle while the KLX ops package is loading.
        from .sub import sub

        return sub(self, other, alpha=alpha)

    if self_is_tensor and not other_is_tensor:
        return _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.sub.Scalar, keyset, self, other, alpha=alpha
        )
    if not self_is_tensor and other_is_tensor:
        return _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.rsub.Scalar, keyset, other, self, alpha=alpha
        )
    if not self_is_tensor and not other_is_tensor:
        self = torch.scalar_tensor(self)
        return _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.sub.Scalar, keyset, self, other, alpha=alpha
        )

    return _redispatch_below_ad_inplace_or_view(
        torch.ops.aten.sub.Tensor, keyset, self, other, alpha=alpha
    )


def _mul_tensor_compat(keyset, self, other):
    self_is_tensor = isinstance(self, torch.Tensor)
    other_is_tensor = isinstance(other, torch.Tensor)

    if _is_fp64_compat_result(self, other):
        from ._fp64_compat import mul_fp64

        return mul_fp64(self, other)
    if _is_fp64_compat_result(other, self):
        from ._fp64_compat import mul_fp64

        # Multiplication is commutative; use the actual f64 tensor as the
        # allocation/device anchor without changing result semantics.
        return mul_fp64(other, self)

    klx_tensor = (
        self if _is_klx_tensor(self) else other if _is_klx_tensor(other) else None
    )
    if klx_tensor is not None and (
        not (self_is_tensor and other_is_tensor)
        or _is_complex_operand(self)
        or _is_complex_operand(other)
    ):
        if isinstance(self, complex):
            self = _complex_scalar_tensor(klx_tensor, self)
        if isinstance(other, complex):
            other = _complex_scalar_tensor(klx_tensor, other)
        # Import lazily to avoid a cycle while the KLX ops package is loading.
        from .mul import mul

        return mul(self, other)

    if self_is_tensor and not other_is_tensor:
        return _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.mul.Scalar, keyset, self, other
        )
    if not self_is_tensor and other_is_tensor:
        return _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.mul.Scalar, keyset, other, self
        )
    if not self_is_tensor and not other_is_tensor:
        self = torch.scalar_tensor(self)
        return _redispatch_below_ad_inplace_or_view(
            torch.ops.aten.mul.Scalar, keyset, self, other
        )

    return _redispatch_below_ad_inplace_or_view(
        torch.ops.aten.mul.Tensor, keyset, self, other
    )


def _rand_fp64_compat(
    keyset,
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    if (
        dtype == torch.float64
        and device is not None
        and torch.device(device).type == "cuda"
    ):
        # Import lazily so ordinary CPU/float32 factories stay native and this
        # module does not create an ops-package import cycle.
        from .rand import rand

        return rand(
            size,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    return torch.ops.aten.rand.default.redispatch(
        keyset - _BACKEND_SELECT_KEYSET,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


def _same_klx_device(source, requested_device):
    if source.device.type != "cuda":
        return False
    if requested_device is None:
        return True
    target = torch.device(requested_device)
    return target.type == "cuda" and (
        target.index is None or target.index == source.device.index
    )


def _to_copy_compat(
    keyset,
    self,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    target_dtype = self.dtype if dtype is None else dtype
    unsupported_pair = (self.dtype, target_dtype) in (
        (torch.bfloat16, torch.int16),
        (torch.int16, torch.bfloat16),
    )
    if (
        unsupported_pair
        and _same_klx_device(self, device)
        and (layout is None or layout == torch.strided)
        and pin_memory is None
    ):
        # Import lazily to avoid a cycle while the KLX ops package is loading.
        from .to import to_copy

        return to_copy(
            self,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )
    return _redispatch_below_ad_inplace_or_view(
        torch.ops.aten._to_copy.default,
        keyset,
        self,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )


# ADInplaceOrView sits above the generated autograd kernel.  Registering here
# lets the ordinary path redispatch through native autograd before reaching the
# active CUDA implementation.  Consequently, a use_gems() CUDA registration
# still takes precedence at the backend key, while the three explicitly broken
# native dtype paths can enter their KLX implementations directly.
_native_compat_lib = torch.library.Library("aten", "IMPL", "ADInplaceOrView")
_native_compat_lib.impl("add.Tensor", _add_tensor_compat, with_keyset=True)
_native_compat_lib.impl("sub.Tensor", _sub_tensor_compat, with_keyset=True)
_native_compat_lib.impl("mul.Tensor", _mul_tensor_compat, with_keyset=True)
_native_compat_lib.impl("_to_copy", _to_copy_compat, with_keyset=True)

# Factory operators have no Tensor argument, so initial dispatch passes through
# BackendSelect and does not include AutogradCUDA.  This key also stays above
# use_gems()' scoped CUDA override, allowing non-f64 requests to redispatch to
# whichever native or scoped CUDA rand kernel is currently active.
_factory_compat_lib = torch.library.Library("aten", "IMPL", "BackendSelect")
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=(
            r"Warning only once for all operators,  other operators may also "
            r"be overridden\."
        ),
        category=UserWarning,
        module=r"torch\.library",
    )
    _factory_compat_lib.impl("rand", _rand_fp64_compat, with_keyset=True)
