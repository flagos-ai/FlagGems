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

"""NVIDIA native baseline for `native_batch_norm_legit_out`.

Mirrors benchmark/test_native_batch_norm_legit.py (NormBenchmark, a
GenericBenchmark subclass; the test dispatches through the local
_run_benchmark helper):
    op_name  = "native_batch_norm_legit_out"
    torch_op = torch.ops.aten._native_batch_norm_legit.out
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:NormBenchmark (benchmark/core_shapes.yaml),
               shape_desc "N, C, *"

The saved statistics are float32 for half/bfloat16 inputs and follow the input
dtype otherwise; the kunlunxin branch that keeps them in the input dtype does
not apply here. training is True, momentum 0.1 and eps 1e-5.

The three outputs (out, save_mean, save_invstd) are passed as one keyword dict,
exactly as the input_fn yields them. This overload also takes
running_mean/running_var (zeros/ones).
"""

import torch

OP_NAME = "native_batch_norm_legit_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("out", "save_mean", "save_invstd")
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._native_batch_norm_legit.out is unavailable on this torch build"
)

# yaml:NormBenchmark -- shape_desc "N, C, *"
_SHAPES = [
    (4, 16, 64, 4),
    (16, 16, 8, 48),
    (16, 16, 8, 88),
    (16, 16, 128),
    (20, 6, 65536),
]

_TRAINING = True
_MOMENTUM = 0.1
_EPS = 1e-5


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
    op = _resolve("torch.ops.aten._native_batch_norm_legit.out")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def _stats_dtype(dtype):
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(channels, dtype=dtype, device=device)
    bias = torch.randn(channels, dtype=dtype, device=device)
    stats_dtype = _stats_dtype(dtype)
    outputs = {
        "out": torch.empty_like(inp),
        "save_mean": torch.empty(channels, dtype=stats_dtype, device=device),
        "save_invstd": torch.empty(channels, dtype=stats_dtype, device=device),
    }
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    return (
        inp,
        weight,
        bias,
        running_mean,
        running_var,
        _TRAINING,
        _MOMENTUM,
        _EPS,
    ), outputs


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    c = shape[1]
    dt = str(dtype)
    sdt = str(_stats_dtype(dtype))
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": dt, "note": "running_mean"},
                {"shape": [c], "dtype": dt, "note": "running_var"},
                {"scalar": _TRAINING},
                {"scalar": _MOMENTUM},
                {"scalar": _EPS},
            ],
            "kwargs": {
                "out": {"shape": shape, "dtype": dt},
                "save_mean": {"shape": [c], "dtype": sdt},
                "save_invstd": {"shape": [c], "dtype": sdt},
            },
        },
        "outputs": {
            "out": {"shape": shape, "dtype": dt},
            "save_mean": {"shape": [c], "dtype": sdt},
            "save_invstd": {"shape": [c], "dtype": sdt},
        },
        "dims": {"N": shape[0], "C": c, "shape": shape},
    }
