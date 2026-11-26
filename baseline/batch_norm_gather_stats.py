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

"""NVIDIA native baseline for `batch_norm_gather_stats`.

Mirrors benchmark/test_batch_norm_gather_stats.py (NormBenchmark, a
GenericBenchmark subclass):
    op_name  = "batch_norm_gather_stats"
    torch_op = torch.batch_norm_gather_stats
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:NormBenchmark (benchmark/core_shapes.yaml)

The torch-composed `_substituted_batch_norm_gather_stats` in that file is
selected only on kunlunxin; on NVIDIA the benchmark uses the real ATen
operator, which is what this baseline resolves.

The input_fn models distributed training with 4 GPU segments: mean and invstd
are [4, C] float32, count is N * prod(shape[2:]), momentum 0.1 and eps 1e-5.
running_mean/running_var are updated in place by the operator.
"""

import torch

OP_NAME = "batch_norm_gather_stats"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (3, 4)
NATIVE_UNAVAILABLE_REASON = (
    "torch.batch_norm_gather_stats is unavailable on this torch build"
)

# yaml:NormBenchmark -- shape_desc "N, C, *"
_SHAPES = [
    (4, 16, 64, 4),
    (16, 16, 8, 48),
    (16, 16, 8, 88),
    (16, 16, 128),
    (20, 6, 65536),
]

# Typical distributed training uses 4 GPU segments.
_NUM_SEGMENTS = 4
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
    op = _resolve("torch.batch_norm_gather_stats")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    n, c = shape[0], shape[1]
    spatial = 1
    for s in shape[2:]:
        spatial *= s
    count = n * spatial

    input_t = torch.randn(shape, dtype=dtype, device=device)
    mean = torch.randn(_NUM_SEGMENTS, c, dtype=torch.float32, device=device)
    invstd = torch.rand(_NUM_SEGMENTS, c, dtype=torch.float32, device=device) + 0.1
    running_mean = torch.randn(c, dtype=torch.float32, device=device)
    running_var = torch.rand(c, dtype=torch.float32, device=device) + 0.1
    return (
        input_t,
        mean,
        invstd,
        running_mean,
        running_var,
        _MOMENTUM,
        _EPS,
        count,
    ), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    n, c = shape[0], shape[1]
    spatial = 1
    for s in shape[2:]:
        spatial *= s
    dt = str(dtype)
    f32 = "torch.float32"
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": [_NUM_SEGMENTS, c], "dtype": f32},
                {"shape": [_NUM_SEGMENTS, c], "dtype": f32},
                {"shape": [c], "dtype": f32, "note": "updated in place"},
                {"shape": [c], "dtype": f32, "note": "updated in place"},
                {"scalar": _MOMENTUM},
                {"scalar": _EPS},
                {"scalar": n * spatial},
            ],
            "kwargs": {},
        },
        "outputs": {
            "mean": {"shape": [c], "dtype": f32},
            "invstd": {"shape": [c], "dtype": f32},
        },
        "dims": {
            "N": n,
            "C": c,
            "spatial": spatial,
            "count": n * spatial,
            "num_segments": _NUM_SEGMENTS,
        },
    }
