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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(16, 64), (16, 128), (32, 128), (64, 256)]
INNER_K_TILES = (2, 4, 8)


def _reference_int4pack(inp, inner_k_tiles):
    del inner_k_tiles
    low = inp[:, 0::2].to(torch.uint8) & 0xF
    high = (inp[:, 1::2].to(torch.uint8) & 0xF) << 4
    return low | high


class ConvertWeightBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "N, K"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for inner_k_tiles in INNER_K_TILES:
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims)},
            params={"innerKTiles": inner_k_tiles},
            builder_args=(dims, inner_k_tiles),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, inner_k_tiles = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randint(0, 16, shape, dtype=torch.int32, device=device)
    return inp, inner_k_tiles


@pytest.mark.convert_weight_to_int4pack
@pytest.mark.underscore_convert_weight_to_int4pack
def test_convert_weight_to_int4pack():
    ConvertWeightBenchmark(
        op_name="_convert_weight_to_int4pack",
        torch_op=_reference_int4pack,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.int32],
    ).run()
