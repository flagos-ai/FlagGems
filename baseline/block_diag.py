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

"""NVIDIA native baseline for `block_diag`.

Mirrors benchmark/test_block_diag.py (BlockDiagBenchmark, a base.Benchmark
subclass):
    op_name  = "block_diag"
    torch_op = _torch_block_diag  (the module-local reference below)
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with BLOCK_DIAG_SHAPES and shape_desc
"num_blocks, block_size", so core_shapes.yaml is not consulted. Each entry
becomes a list of num_blocks square tensors of block_size x block_size, passed
as a single list argument; the benchmark's wrapper unpacks it into
torch.block_diag(*tensors).
"""

import torch

OP_NAME = "block_diag"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.block_diag is unavailable on this torch build"

# test_block_diag.py::BLOCK_DIAG_SHAPES -- (num_blocks, block_size)
_SHAPES = [
    (4, 64),
    (8, 128),
    (16, 64),
    (4, 256),
    (8, 256),
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


def _torch_block_diag(tensors):
    """benchmark/test_block_diag.py::_torch_block_diag -- list in, varargs out."""
    return torch.block_diag(*tensors)


def native():
    if _resolve("torch.block_diag") is None:
        return None
    return _torch_block_diag


def grid():
    return [
        {"num_blocks": num_blocks, "block_size": block_size}
        for num_blocks, block_size in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    num_blocks, block_size = binding["num_blocks"], binding["block_size"]
    blocks = [
        torch.randn((block_size, block_size), dtype=dtype, device=device)
        for _ in range(num_blocks)
    ]
    return (blocks,), {}


def key_shape(binding):
    return [binding["num_blocks"], binding["block_size"]]


def config(binding, dtype):
    num_blocks, block_size = binding["num_blocks"], binding["block_size"]
    dt = str(dtype)
    side = num_blocks * block_size
    return {
        "inputs": {
            "args": [
                [
                    {"shape": [block_size, block_size], "dtype": dt}
                    for _ in range(num_blocks)
                ]
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [side, side], "dtype": dt}},
        "dims": {"num_blocks": num_blocks, "block_size": block_size},
    }
