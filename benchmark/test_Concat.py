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

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0}


class ConcatBenchmark(base.Benchmark):
    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)

    def set_more_shapes(self):
        more_shapes_2d = [(1024, 2**i) for i in range(1, 11, 4)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 8, 4)]
        return more_shapes_2d + more_shapes_3d

    def record_shapes(self, *args, **kwargs):
        # Override to produce a flat list of tensor shapes instead of a
        # (args, kwargs) tuple, which would produce parentheses that are
        # incompatible with the benchmark output parser.
        def deep_parse(item):
            if isinstance(item, torch.Tensor):
                return item.size()
            elif isinstance(item, (int, float, str, torch.dtype)):
                return item
            elif isinstance(item, (list, tuple)):
                return [deep_parse(sub_item) for sub_item in item]
            elif isinstance(item, dict):
                return {key: deep_parse(value) for key, value in item.items()}
            return None

        parsed = [deep_parse(arg) for arg in args]
        # Return just the parsed list (no kwargs wrapper) so the outermost
        # bracket is '[' instead of '('.
        return parsed


@pytest.mark.concat
def test_Concat():
    bench = ConcatBenchmark(
        input_fn=_input_fn,
        op_name="concat",
        torch_op=torch.cat,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
