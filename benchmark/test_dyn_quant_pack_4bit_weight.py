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

from typing import Generator

import pytest
import torch

from flag_gems.ops._dyn_quant_pack_4bit_weight import _dyn_quant_pack_4bit_weight

from . import base


def torch_pack_reference(
    weights, scales_zeros, bias, block_size, in_features, out_features
):
    del block_size, in_features, out_features
    parts = [weights.flatten().float(), scales_zeros.flatten().float()]
    if bias is not None:
        parts.append(bias.flatten().float())
    return torch.cat(parts)


class DynQuantPack4BitWeightBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Representative linear-layer widths with full-width and grouped quantization.
        self.shapes = [(64, 64, 32), (256, 256, 64), (1024, 1024, 128)]
        self.shape_desc = "OUT_FEATURES, IN_FEATURES, BLOCK_SIZE"

    def get_input_iter(self, cur_dtype) -> Generator:
        for out_features, in_features, block_size in self.shapes:
            weights = torch.randint(
                0,
                256,
                (out_features, in_features // 2),
                dtype=torch.uint8,
                device=self.device,
            )
            groups = in_features // block_size
            scales = torch.randn(
                (out_features, groups), dtype=cur_dtype, device=self.device
            )
            bias = torch.randn(out_features, dtype=cur_dtype, device=self.device)
            yield weights, scales, bias, block_size, in_features, out_features


@pytest.mark.dyn_quant_pack_4bit_weight
def test_dyn_quant_pack_4bit_weight():
    bench = DynQuantPack4BitWeightBenchmark(
        op_name="dyn_quant_pack_4bit_weight",
        torch_op=torch_pack_reference,
        # The portable packed representation stores float32 scale/zero and bias values.
        dtypes=[torch.float32],
    )
    bench.set_gems(_dyn_quant_pack_4bit_weight)
    bench.run()
