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


def fake_quantize_per_tensor_affine_cachemask_input_fn(shape, dtype, device):
    input = utils.generate_tensor_input(shape, dtype, device)
    yield input, 0.125, 3, 0, 255


def fake_quantize_per_tensor_affine_cachemask_out_input_fn(shape, dtype, device):
    input = utils.generate_tensor_input(shape, dtype, device)
    out0 = torch.empty_like(input)
    out1 = torch.empty_like(input, dtype=torch.bool)
    yield input, 0.125, 3, 0, 255, {"out0": out0, "out1": out1}


class BenchmarkFakeQuantizePerTensorAffineCachemask(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 4),
            (64, 64),
            (128, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (8192, 8192),
            (2, 3, 128, 128),
            (8, 16, 64, 64),
            (32, 64, 128, 128),
        ]


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
def test_fake_quantize_per_tensor_affine_cachemask():

    bench = BenchmarkFakeQuantizePerTensorAffineCachemask(
        op_name="fake_quantize_per_tensor_affine_cachemask",
        torch_op=torch.ops.aten.fake_quantize_per_tensor_affine_cachemask,
        input_fn=fake_quantize_per_tensor_affine_cachemask_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_out
def test_fake_quantize_per_tensor_affine_cachemask_out():
    bench = BenchmarkFakeQuantizePerTensorAffineCachemask(
        op_name="fake_quantize_per_tensor_affine_cachemask_out",
        torch_op=torch.ops.aten.fake_quantize_per_tensor_affine_cachemask.out,
        input_fn=fake_quantize_per_tensor_affine_cachemask_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
