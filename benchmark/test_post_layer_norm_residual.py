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

CASES = [
    ((64, 128), (128,)),
    ((256, 512), (512,)),
    ((1024, 512), (512,)),
    ((2048, 512), (512,)),
    ((16, 32, 16, 32), (16, 32)),
    ((1024, 513), (513,)),
]


def torch_op(input, residual, normalized_shape, weight, bias, eps):
    return torch.layer_norm(input, normalized_shape, weight, bias, eps) + residual


def torch_training_op(
    input, residual, normalized_shape, weight, bias, eps, grad_output
):
    output = torch_op(input, residual, normalized_shape, weight, bias, eps)
    return torch.autograd.grad(output, (input, residual, weight, bias), grad_output)


def gems_training_op(input, residual, normalized_shape, weight, bias, eps, grad_output):
    output = flag_gems.post_layer_norm_residual(
        input, residual, normalized_shape, weight, bias, eps
    )
    return torch.autograd.grad(output, (input, residual, weight, bias), grad_output)


class PostLayerNormResidualBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input_shape, normalized_shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = CASES

    def get_input_iter(self, dtype):
        for shape, normalized_shape in self.shapes:
            input = torch.randn(shape, dtype=dtype, device=self.device)
            residual = torch.randn_like(input)
            weight = torch.randn(normalized_shape, dtype=dtype, device=self.device)
            bias = torch.randn_like(weight)
            yield input, residual, normalized_shape, weight, bias, 1e-5


class PostLayerNormResidualTrainingBenchmark(PostLayerNormResidualBenchmark):
    def get_input_iter(self, dtype):
        for shape, normalized_shape in self.shapes:
            input = torch.randn(
                shape, dtype=dtype, device=self.device, requires_grad=True
            )
            residual = torch.randn_like(input, requires_grad=True)
            weight = torch.randn(
                normalized_shape,
                dtype=dtype,
                device=self.device,
                requires_grad=True,
            )
            bias = torch.randn_like(weight, requires_grad=True)
            grad_output = torch.randn_like(input)
            yield input, residual, normalized_shape, weight, bias, 1e-5, grad_output


@pytest.mark.post_layer_norm_residual
def test_post_layer_norm_residual():
    bench = PostLayerNormResidualBenchmark(
        op_name="post_layer_norm_residual",
        torch_op=torch_op,
        gems_op=flag_gems.post_layer_norm_residual,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.post_layer_norm_residual
def test_post_layer_norm_residual_training(monkeypatch):
    if flag_gems.vendor_name == "mthreads":
        monkeypatch.setenv("DISABLE_LLVM_OPT", "1")
    bench = PostLayerNormResidualTrainingBenchmark(
        op_name="post_layer_norm_residual_training",
        torch_op=torch_training_op,
        gems_op=gems_training_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
