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

from . import base
from .generated_operator_utils import OperatorBenchmark

SCALE = 0.1
ZERO_POINT = 0
QDTYPE = torch.quint8


def _quantized_device(value, device):
    if flag_gems.vendor_name == "ascend" and torch.device(device).type != "cpu":
        result = torch._empty_affine_quantized(
            value.shape,
            scale=value.q_scale(),
            zero_point=value.q_zero_point(),
            dtype=value.dtype,
            device=device,
        )
        result.untyped_storage().copy_(value.untyped_storage())
        return result
    return value.to(device)


def _make_quantized(shape, device):
    torch.manual_seed(0)
    host = torch.quantize_per_tensor(
        torch.randn(shape).clamp_(-2, 2), SCALE, ZERO_POINT, QDTYPE
    )
    result = _quantized_device(host, device)
    result._cpu_twin = host
    return result


def _cpu_input(value):
    twin = getattr(value, "_cpu_twin", None)
    return twin if twin is not None else value.to("cpu")


SHAPES = [(4, 1024), (16, 4096), (32, 8192), (64, 16384), (8, 3, 1024), (16, 8, 2048)]
PARAMS = {
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
    "dilation": 1,
    "ceil_mode": False,
}


class QuantizedMaxPool1dBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "N[, C], L"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    out_l = (
        dims[-1]
        + 2 * PARAMS["padding"]
        - (PARAMS["kernel_size"] - 1) * PARAMS["dilation"]
        - 1
    ) // PARAMS["stride"] + 1
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "out": list(dims[:-1] + (out_l,))},
        params=dict(PARAMS),
        builder_args=(dims, out_l),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, _ = plan.builder_args
    return _make_quantized(shape, device), dict(PARAMS)


def _torch_op(qx, **kwargs):
    return torch.quantized_max_pool1d(_cpu_input(qx), **kwargs)


@pytest.mark.quantized_max_pool1d
def test_quantized_max_pool1d():
    QuantizedMaxPool1dBenchmark(
        op_name="quantized_max_pool1d",
        torch_op=_torch_op,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[QDTYPE],
    ).run()
