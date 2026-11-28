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

"""NVIDIA native baseline for `sparse_broadcast_to`.

Mirrors benchmark/test_sparse_broadcast_to.py (SparseBroadcastToBenchmark, a
base.Benchmark subclass):
    op_name  = "sparse_broadcast_to"
    torch_op = torch.ops.aten._sparse_broadcast_to
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own source-shape list and declares
shape_desc "src_shape -> target_shape", so core_shapes.yaml is not consulted.
The target prepends a new sparse dimension of size 4 to the source shape.

The input is a coalesced COO tensor at 30% density, built as
(randn * (rand < density)).to_sparse().coalesce().
"""

import torch

OP_NAME = "sparse_broadcast_to"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._sparse_broadcast_to is unavailable on this torch build"
)

# test_sparse_broadcast_to.py::SPARSE_BROADCAST_TO_SHAPES
_SHAPES = [
    (256,),
    (20, 320, 15),
    (16, 128, 64, 60),
]

_DENSITY = 0.3
_PREPEND = 4


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


def _make_sparse_coo(shape, dtype, device, density=_DENSITY):
    """benchmark::_make_sparse_coo -- coalesced COO on the target device."""
    x = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand(shape, device=device) < density
    return (x * mask).to_sparse().coalesce()


def native():
    op = _resolve("torch.ops.aten._sparse_broadcast_to")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    x = _make_sparse_coo(shape, dtype, device)
    # Prepend a new sparse dimension of size 4.
    return (x, (_PREPEND,) + shape), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    target = [_PREPEND] + shape
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {
                    "shape": shape,
                    "dtype": dt,
                    "layout": "torch.sparse_coo",
                    "density": _DENSITY,
                },
                target,
            ],
            "kwargs": {},
        },
        "outputs": {
            "out": {
                "shape": target,
                "dtype": dt,
                "layout": "torch.sparse_coo",
            }
        },
        "dims": {"src_shape": shape, "target_shape": target},
    }
