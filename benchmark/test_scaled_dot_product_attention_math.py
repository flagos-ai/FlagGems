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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(2, 4, 64, 64), (2, 8, 128, 64), (4, 8, 256, 64)]


class AttentionBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "(batch, heads, sequence, head_dim)"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for is_causal in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"query": list(dims), "key": list(dims), "value": list(dims)},
            params={"is_causal": is_causal},
            builder_args=(dims, is_causal),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, is_causal = plan.builder_args
    torch.manual_seed(0)
    query = torch.randn(shape, dtype=dtype, device=device)
    key = torch.randn(shape, dtype=dtype, device=device)
    value = torch.randn(shape, dtype=dtype, device=device)
    return query, key, value, {"is_causal": is_causal}


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.underscore_scaled_dot_product_attention_math
def test_scaled_dot_product_attention_math():
    AttentionBenchmark(
        op_name="_scaled_dot_product_attention_math",
        torch_op=torch.ops.aten._scaled_dot_product_attention_math,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
