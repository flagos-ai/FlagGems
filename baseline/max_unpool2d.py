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

"""NVIDIA native baseline for `max_unpool2d`.

Mirrors benchmark/test_max_unpool2d.py (MaxUnpool2dBenchmark, a base.Benchmark
subclass):
    op_name  = "max_unpool2d"
    torch_op = torch_max_unpool2d, a pass-through to
               torch.ops.aten.max_unpool2d
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with MAX_UNPOOL2D_SHAPES and returns None from
set_more_shapes, so core_shapes.yaml is not consulted.

pooled and indices come from a real MaxPool2d(2, stride=2,
return_indices=True) forward, which build_inputs performs outside the timed
region as the benchmark's get_input_iter does; output_size is the original
[H, W] and the indices are cast to int64.
"""

import torch

OP_NAME = "max_unpool2d"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.max_unpool2d is unavailable on this torch build"
)

# test_max_unpool2d.py::MAX_UNPOOL2D_SHAPES -- (N, C, H, W)
_SHAPES = [
    (1, 1, 8, 8),
    (1, 1, 16, 16),
    (2, 3, 16, 16),
    (4, 8, 32, 32),
    (1, 16, 32, 32),
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


def _torch_max_unpool2d(pooled, indices, output_size):
    """benchmark::test_max_unpool2d::torch_max_unpool2d."""
    return torch.ops.aten.max_unpool2d(pooled, indices, output_size)


def native():
    if _resolve("torch.ops.aten.max_unpool2d") is None:
        return None
    return _torch_max_unpool2d


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    _, _, h, w = shape
    x = torch.randn(shape, dtype=dtype, device=device)
    pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
    pooled, indices = pool(x.contiguous())
    return (pooled, indices.to(torch.int64), [h, w]), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    n, c, h, w = shape
    pooled_shape = [n, c, h // 2, w // 2]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": pooled_shape, "dtype": dt},
                {"shape": pooled_shape, "dtype": "torch.int64"},
                [h, w],
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"N": n, "C": c, "H": h, "W": w},
    }
