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

"""NVIDIA baseline for log_sigmoid_backward (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.ops.aten.log_sigmoid_backward``, the ATen kernel the operator is a port
of; the ``grad_output * torch.sigmoid(-inp)`` composition that FlagGems falls
back to on Ascend builds without that kernel is not needed here, since
baselines are collected on NVIDIA. Shapes (after the benchmark's shape
replacements and >= 1M-element filter), dtypes and the
``(grad_output, input, buffer=exp(-abs(input)))`` input construction are copied
from ``benchmark/test_log_sigmoid_backward.py``.
"""

import torch

OP_NAME = "log_sigmoid_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/test_log_sigmoid_backward.py::LogSigmoidBackwardBenchmark.set_shapes
# (core_shapes.yaml::UnaryPointwiseBenchmark, then replacements + >=1M filter)
_CASES = [
    (33554432,),
    (4096, 4096),
    (64, 512, 512),
    (128, 512, 512),
]


def native():
    return torch.ops.aten.log_sigmoid_backward


def grid():
    return [{"shape": case} for case in _CASES]


def build_inputs(binding, dtype, device):
    inp = torch.randn(binding["shape"], dtype=dtype, device=device)
    grad_output = torch.randn(binding["shape"], dtype=dtype, device=device)
    buffer = torch.exp(-torch.abs(inp))
    return (grad_output, inp, buffer), {}


def key_shape(binding):
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    # All three operands and the result share the case shape and dtype.
    shape = list(binding["shape"])
    return {
        "inputs": {
            "inp": {"shape": shape, "dtype": str(dtype)},
            "grad_output": {"shape": shape, "dtype": str(dtype)},
            "buffer": {"shape": shape, "dtype": str(dtype)},
        },
        "outputs": {"out": {"shape": shape, "dtype": str(dtype)}},
        "dims": {},
    }
