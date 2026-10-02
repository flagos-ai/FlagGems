# Copyright (c) 2025, FlagGems Contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import base

# q_scale reads the stored per-tensor affine scale of a quantized tensor. Only
# quantized containers carry a quantizer and torch.quantize_per_tensor accepts
# float32 source data only, so that is the sole benchmark dtype. The shared
# default shapes (and a caller --shape_file) drive the suite.

_Q_CASES = [
    (torch.quint8, 0.1, 0),
    (torch.qint8, 0.0078125, 3),
    (torch.qint32, 2.0, 128),
]


def _case_fn(shape, dtype):
    del dtype
    for q_dtype, scale, zero_point in _Q_CASES:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={
                "q_dtype": str(q_dtype),
                "scale": scale,
                "zero_point": zero_point,
            },
            builder_args=(shape, q_dtype, scale, zero_point),
        )


def _build_inputs_fn(plan, dtype, device):
    # The quantizer parameters below are exactly the ones reported in the plan
    # metadata, so a listed case and its executed input agree.
    shape, q_dtype, scale, zero_point = plan.builder_args
    src = torch.rand(shape, dtype=dtype, device=device)
    return torch.quantize_per_tensor(src, scale, zero_point, q_dtype), {}


@pytest.mark.q_scale
def test_q_scale():
    bench = base.GenericBenchmark(
        op_name="q_scale",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.q_scale,
        gems_op=getattr(flag_gems, "q_scale", None),
        dtypes=[torch.float32],
    )
    bench.run()
