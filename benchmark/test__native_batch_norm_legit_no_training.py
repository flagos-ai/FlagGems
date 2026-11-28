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
import torch.nn.functional as F

import flag_gems

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def native_batch_norm_legit_no_training_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = torch.randn((C,), dtype=dtype, device=device)
    running_var = torch.abs(torch.randn((C,), dtype=dtype, device=device)) + 0.1
    momentum = 0.1
    eps = 1e-5
    yield inp, weight, bias, running_mean, running_var, momentum, eps


def torch_native_batch_norm_legit_no_training(
    inp, weight, bias, running_mean, running_var, momentum, eps
):
    return F.batch_norm(
        inp, running_mean, running_var, weight, bias, training=False, eps=eps
    )


@pytest.mark.native_batch_norm_legit_no_training
def test_native_batch_norm_legit_no_training():
    bench = NormBenchmark(
        input_fn=native_batch_norm_legit_no_training_input_fn,
        op_name="native_batch_norm_legit_no_training",
        torch_op=torch_native_batch_norm_legit_no_training,
        gems_op=flag_gems._native_batch_norm_legit_no_training,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def native_batch_norm_legit_no_training_backward_input_fn(shape, dtype, device):
    """Same tensors as the forward benchmark; the graph is built by the framework."""
    for args in native_batch_norm_legit_no_training_input_fn(shape, dtype, device):
        yield args


def _infer_bn_output(inp, weight, bias, running_mean, running_var, momentum, eps):
    """Baseline forward: single output tensor (the framework mimics autograd)."""
    return torch.ops.aten._native_batch_norm_legit_no_training.default(
        inp, weight, bias, running_mean, running_var, momentum, eps
    )[0]


def _infer_bn_output_gems(inp, weight, bias, running_mean, running_var, momentum, eps):
    """Candidate forward through the FlagGems autograd.Function wrapper."""
    return flag_gems._native_batch_norm_legit_no_training(
        inp, weight, bias, running_mean, running_var, momentum, eps
    )[0]


class InferBNBackwardBenchmark(NormBenchmark):
    """Autograd backward of the inference-only batch normalization.

    ``Benchmark.unpack_to_args_kwargs`` marks every floating input as requiring a
    gradient, but ATen rejects ``requires_grad`` on ``running_mean`` /
    ``running_var``: they are constants of the no-training path. Only ``input``,
    ``weight`` and ``bias`` take part in the backward, so they alone are marked.
    """

    def unpack_to_args_kwargs(self, input_tuple):
        args = list(input_tuple)
        for index in range(3):
            if torch.is_tensor(args[index]) and torch.is_floating_point(args[index]):
                args[index].requires_grad_(True)
        return args, {}


def test_native_batch_norm_legit_no_training_backward():
    """Timed ``torch.autograd.grad`` on the default ``NormBenchmark`` shapes.

    Baseline: native ATen forward + engine ``native_batch_norm_backward``
    (train=False). Candidate: ``flag_gems._native_batch_norm_legit_no_training``
    (autograd.Function) + ``flag_gems.batch_norm_backward`` (train=False).
    """
    bench = InferBNBackwardBenchmark(
        input_fn=native_batch_norm_legit_no_training_backward_input_fn,
        op_name="native_batch_norm_legit_no_training",
        torch_op=_infer_bn_output,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(_infer_bn_output_gems)
    bench.run()
