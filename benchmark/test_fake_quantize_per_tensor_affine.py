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

from . import base, consts, utils


def fake_quantize_per_tensor_affine_input_fn(shape, dtype, device):
    input = utils.generate_tensor_input(shape, dtype, device)
    yield input, 0.125, 3, 0, 255


@pytest.mark.fake_quantize_per_tensor_affine
def test_fake_quantize_per_tensor_affine():
    class BenchmarkFakeQuantizePerTensorAffine(base.GenericBenchmark):
        def set_shapes(self, shape_file_path=None):
            self.shapes = [
                (4, 4),
                (64, 64),
                (128, 256),
                (512, 512),
                (1024, 1024),
                (2, 3, 128, 128),
                (8, 16, 64, 64),
            ]

    bench = BenchmarkFakeQuantizePerTensorAffine(
        op_name="fake_quantize_per_tensor_affine",
        torch_op=torch.fake_quantize_per_tensor_affine,
        input_fn=fake_quantize_per_tensor_affine_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
