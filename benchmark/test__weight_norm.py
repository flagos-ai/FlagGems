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

import pytest
import torch

import flag_gems

from . import base, consts


def weight_norm_input_fn(shape, dtype, device):
    dim = 0
    v = torch.randn(shape, dtype=dtype, device=device)
    g = torch.randn(
        [1 if i != dim else shape[i] for i in range(len(shape))],
        dtype=dtype,
        device=device,
    )
    yield v, g, dim


def weight_norm_input_fn_last(shape, dtype, device):
    dim = len(shape) - 1
    v = torch.randn(shape, dtype=dtype, device=device)
    g = torch.randn(
        [1 if i != dim else shape[i] for i in range(len(shape))],
        dtype=dtype,
        device=device,
    )
    yield v, g, dim


# ``_weight_norm`` is a distinct operator from ``weight_norm``. Its id cannot be
# used verbatim as a pytest marker, for two independent reasons:
#
#  1. pytest rejects the attribute form outright: MarkGenerator.__getattr__
#     raises ``AttributeError("Marker name must NOT start with underscore")``.
#  2. check_operator_markers derives the required marker from the operator id by
#     stripping the leading underscore and -- because ``weight_norm`` is already
#     a registered operator id, so the stripped name would collide -- prefixing
#     ``underscore_``, giving ``underscore_weight_norm``. Its AST scan accepts
#     only a literal ``pytest.mark.<that name>`` attribute, so hand-building a
#     MarkDecorator to dodge (1) would read as "no marker" and fail the gate.
#
# Hence ``underscore_weight_norm``, mirroring op_marker() in tools/run_tests.py.
@pytest.mark.underscore_weight_norm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_underscore_weight_norm_dim0():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="_weight_norm",
        input_fn=weight_norm_input_fn,
        torch_op=torch.ops.aten._weight_norm.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.underscore_weight_norm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_underscore_weight_norm_dim_last():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="_weight_norm",
        input_fn=weight_norm_input_fn_last,
        torch_op=torch.ops.aten._weight_norm.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
