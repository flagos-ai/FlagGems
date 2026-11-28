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

"""NVIDIA native baseline for `batch_norm_impl_index_backward`.

Mirrors benchmark/test_batch_norm_impl_index_backward.py (NormBenchmark, a
GenericBenchmark subclass):
    op_name  = "batch_norm_impl_index_backward"
    torch_op = torch.ops.aten._batch_norm_impl_index_backward
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:NormBenchmark (benchmark/core_shapes.yaml)

Schema order is (impl_index, input, grad_output, weight, running_mean,
running_var, save_mean, save_var_transform, train, eps, output_mask,
reservedSpace). impl_index 0 selects the native backend, which supports every
floating dtype and mirrors the CUDA dispatcher's `!train or impl_index == 0`
fallback. save_mean/save_var are float32 (the accumulate type the native
forward produces) and save_var holds the inverse standard deviation.
"""

import torch

OP_NAME = "batch_norm_impl_index_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._batch_norm_impl_index_backward is unavailable on this build"
)

# yaml:NormBenchmark -- shape_desc "N, C, *"
_SHAPES = [
    (4, 16, 64, 4),
    (16, 16, 8, 48),
    (16, 16, 8, 88),
    (16, 16, 128),
    (20, 6, 65536),
]

_IMPL_INDEX = 0
_TRAIN = True
_EPS = 1e-5
_OUTPUT_MASK = [True, True, True]


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
    op = _resolve("torch.ops.aten._batch_norm_impl_index_backward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    c = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn_like(inp)
    weight = torch.randn((c,), dtype=dtype, device=device)
    running_mean = torch.zeros((c,), dtype=dtype, device=device)
    running_var = torch.ones((c,), dtype=dtype, device=device)
    save_mean = torch.randn((c,), dtype=torch.float32, device=device)
    save_var = torch.abs(torch.randn((c,), dtype=torch.float32, device=device)) + 0.1
    reserved_space = torch.empty(0, dtype=torch.uint8, device=device)
    return (
        _IMPL_INDEX,
        inp,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        _TRAIN,
        _EPS,
        _OUTPUT_MASK,
        reserved_space,
    ), {}


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
                {"scalar": _IMPL_INDEX},
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": f32},
                {"shape": [c], "dtype": f32},
                {"scalar": _TRAIN},
                {"scalar": _EPS},
                list(_OUTPUT_MASK),
                {"shape": [0], "dtype": "torch.uint8"},
            ],
            "kwargs": {},
        },
        "outputs": {
            "grad_input": {"shape": shape, "dtype": dt},
            "grad_weight": {"shape": [c], "dtype": dt},
            "grad_bias": {"shape": [c], "dtype": dt},
        },
        "dims": {"N": shape[0], "C": c, "shape": shape},
    }
