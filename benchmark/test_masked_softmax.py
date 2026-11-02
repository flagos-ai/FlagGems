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


class MaskedSoftmaxBenchmark(base.Benchmark):
    # Softmax reduces over the last dim, which one Triton block must span, so the
    # generic pointwise DEFAULT_SHAPES (1D 1G-element / huge trailing dims) do
    # not apply. Pin reduction-friendly 2D shapes and override set_shapes so CI
    # cannot inject oversized pointwise shapes (skill Rule 14).
    DEFAULT_SHAPES = [(64, 64), (1024, 1024), (4096, 4096), (8192, 2048)]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        # mask_type 2: elementwise mask, same shape as the input.
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            mask = torch.randint(0, 2, shape, dtype=torch.bool, device=self.device)
            yield inp, mask, -1, 2

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


@pytest.mark.masked_softmax
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_masked_softmax(dtype):
    torch_op = (
        lambda inp, mask, dim, mask_type: torch.ops.aten._masked_softmax(  # noqa: E731
            inp, mask, dim, mask_type
        )
    )
    bench = MaskedSoftmaxBenchmark(
        op_name="masked_softmax",
        torch_op=torch_op,
        dtypes=[dtype],
    )
    bench.run()
