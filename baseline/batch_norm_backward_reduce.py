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

"""NVIDIA native baseline for `batch_norm_backward_reduce`.

Mirrors benchmark/test_batch_norm_backward_reduce_perf.py (NormBenchmark, a
GenericBenchmark subclass):
    op_name  = "batch_norm_backward_reduce"
    torch_op = torch.batch_norm_backward_reduce
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:NormBenchmark (benchmark/core_shapes.yaml)

The torch-composed `_substituted_batch_norm_backward_reduce` in that file is
selected only on kunlunxin; on NVIDIA the benchmark uses the real ATen
operator, which is what this baseline resolves.

mean and invstd are always float32 even for half inputs, and all three
input_g/weight_g/bias_g flags are True so every output is produced.
"""

import torch

OP_NAME = "batch_norm_backward_reduce"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.batch_norm_backward_reduce is unavailable on this torch build"
)

# yaml:NormBenchmark -- shape_desc "N, C, *"
_SHAPES = [
    (4, 16, 64, 4),
    (16, 16, 8, 48),
    (16, 16, 8, 88),
    (16, 16, 128),
    (20, 6, 65536),
]


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def native():
    op = _resolve("torch.batch_norm_backward_reduce")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    c = shape[1]
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    # mean and invstd must always be float32, even when input is float16.
    mean = torch.randn(c, dtype=torch.float32, device=device)
    invstd = torch.randn(c, dtype=torch.float32, device=device).abs() + 0.1
    weight = torch.randn(c, dtype=dtype, device=device)
    return (grad_output, inp, mean, invstd, weight, True, True, True), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    c = shape[1]
    dt = str(dtype)
    f32 = "torch.float32"
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": dt},
                {"shape": [c], "dtype": f32},
                {"shape": [c], "dtype": f32},
                {"shape": [c], "dtype": dt},
                True,
                True,
                True,
            ],
            "kwargs": {},
        },
        "outputs": {
            "sum_dy": {"shape": [c], "dtype": f32},
            "sum_dy_xmu": {"shape": [c], "dtype": f32},
            "grad_weight": {"shape": [c], "dtype": dt},
            "grad_bias": {"shape": [c], "dtype": dt},
        },
        "dims": {"N": shape[0], "C": c, "shape": shape},
    }
