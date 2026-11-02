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

import warnings

import pytest
import torch

import flag_gems

from . import base

_ZETA_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _ZETA_DTYPES.append(torch.float64)

_ASCEND_NATIVE_SCHEMAS = {
    "special_zeta": "aten::special_zeta",
    "special_zeta_out": "aten::special_zeta.out",
    "special_zeta_tensor_scalar": "aten::special_zeta.other_scalar",
    "special_zeta_tensor_scalar_out": "aten::special_zeta.other_scalar_out",
    "special_zeta_scalar_tensor": "aten::special_zeta.self_scalar",
    "special_zeta_scalar_tensor_out": "aten::special_zeta.self_scalar_out",
}


class _SpecialZetaBenchmark(base.GenericBenchmark):
    """Skip a dtype/overload when its native baseline falls back to CPU."""

    def _probe_native(self, dtype):
        if flag_gems.vendor_name == "ascend":
            schema = _ASCEND_NATIVE_SCHEMAS[self.op_name]
            try:
                has_native_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(
                    schema, "PrivateUse1"
                )
            except (AttributeError, RuntimeError):
                has_native_kernel = False
            if not has_native_kernel:
                pytest.skip(
                    f"native {schema} has no Ascend PrivateUse1 kernel and "
                    "falls back to CPU"
                )

        probe = next(self.input_fn((4,), dtype, self.device))
        args, kwargs = self.unpack_to_args_kwargs(probe)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                self.torch_op(*args, **kwargs)
                flag_gems.runtime.torch_device_fn.synchronize()
            except RuntimeError as error:
                message = str(error).lower()
                if "not implemented" in message or "not support" in message:
                    pytest.skip(
                        f"native {self.op_name} baseline is unavailable for {dtype}: "
                        f"{error}"
                    )
                raise

        fallback_messages = [
            str(item.message)
            for item in caught
            if "cpu" in str(item.message).lower()
            and "fallback" in str(item.message).lower()
        ]
        if fallback_messages:
            pytest.skip(
                f"native {self.op_name} {dtype} baseline falls back to CPU: "
                f"{fallback_messages[0]}"
            )

    def get_input_iter(self, dtype):
        self._probe_native(dtype)
        yield from super().get_input_iter(dtype)


def _tensor_tensor_input(shape, dtype, device):
    x = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 1.05
    q = torch.rand(shape, dtype=dtype, device=device) * 6.0 + 0.1
    yield x, q


def _tensor_tensor_out_input(shape, dtype, device):
    x, q = next(_tensor_tensor_input(shape, dtype, device))
    yield x, q, {"out": torch.empty_like(x)}


def _tensor_scalar_input(shape, dtype, device):
    x = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 1.05
    yield x, 2.25


def _tensor_scalar_out_input(shape, dtype, device):
    x, q = next(_tensor_scalar_input(shape, dtype, device))
    yield x, q, {"out": torch.empty_like(x)}


def _scalar_tensor_input(shape, dtype, device):
    q = torch.rand(shape, dtype=dtype, device=device) * 6.0 + 0.1
    yield 3.5, q


def _scalar_tensor_out_input(shape, dtype, device):
    x, q = next(_scalar_tensor_input(shape, dtype, device))
    yield x, q, {"out": torch.empty_like(q)}


@pytest.mark.special_zeta
def test_special_zeta():
    bench = _SpecialZetaBenchmark(
        op_name="special_zeta",
        torch_op=torch.special.zeta,
        input_fn=_tensor_tensor_input,
        dtypes=_ZETA_DTYPES,
    )
    bench.run()


@pytest.mark.special_zeta_out
def test_special_zeta_out():
    bench = _SpecialZetaBenchmark(
        op_name="special_zeta_out",
        torch_op=torch.ops.aten.special_zeta.out,
        input_fn=_tensor_tensor_out_input,
        dtypes=_ZETA_DTYPES,
    )
    bench.run()


@pytest.mark.special_zeta_tensor_scalar
def test_special_zeta_tensor_scalar():
    bench = _SpecialZetaBenchmark(
        op_name="special_zeta_tensor_scalar",
        torch_op=torch.special.zeta,
        input_fn=_tensor_scalar_input,
        dtypes=_ZETA_DTYPES,
    )
    bench.run()


@pytest.mark.special_zeta_tensor_scalar_out
def test_special_zeta_tensor_scalar_out():
    bench = _SpecialZetaBenchmark(
        op_name="special_zeta_tensor_scalar_out",
        torch_op=torch.ops.aten.special_zeta.other_scalar_out,
        input_fn=_tensor_scalar_out_input,
        dtypes=_ZETA_DTYPES,
    )
    bench.run()


@pytest.mark.special_zeta_scalar_tensor
def test_special_zeta_scalar_tensor():
    bench = _SpecialZetaBenchmark(
        op_name="special_zeta_scalar_tensor",
        torch_op=torch.special.zeta,
        input_fn=_scalar_tensor_input,
        dtypes=_ZETA_DTYPES,
    )
    bench.run()


@pytest.mark.special_zeta_scalar_tensor_out
def test_special_zeta_scalar_tensor_out():
    bench = _SpecialZetaBenchmark(
        op_name="special_zeta_scalar_tensor_out",
        torch_op=torch.ops.aten.special_zeta.self_scalar_out,
        input_fn=_scalar_tensor_out_input,
        dtypes=_ZETA_DTYPES,
    )
    bench.run()
