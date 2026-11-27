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

"""NVIDIA native baseline for `sparse_broadcast_to_copy`.

Mirrors benchmark/test_sparse_broadcast_to_copy.py
(SparseBroadcastToCopyBenchmark, a base.Benchmark subclass):
    op_name  = "sparse_broadcast_to_copy"
    torch_op = torch.ops.aten._sparse_broadcast_to
    dtypes   = [torch.float32]

Note the baseline operator: there is no `_sparse_broadcast_to_copy` in ATen, so
the benchmark compares the copy variant against the plain
`aten::_sparse_broadcast_to`. The semantic difference (view vs copy) is not
reflected in the reference. See Q&A.md.

The class overrides set_shapes with explicit (src_shape, target_shape) pairs,
so core_shapes.yaml is not consulted. The input is a coalesced COO tensor at
30% density.
"""

import torch

OP_NAME = "sparse_broadcast_to_copy"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._sparse_broadcast_to is unavailable on this torch build"
)

# test_sparse_broadcast_to_copy.py::SPARSE_BROADCAST_TO_COPY_SHAPES
# (src_shape, target_shape)
_SHAPES = [
    ((256,), (4, 256)),
    ((20, 320, 15), (4, 20, 320, 15)),
    ((1024, 1024), (4, 1024, 1024)),
]

_DENSITY = 0.3


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
    op = _resolve("torch.ops.aten._sparse_broadcast_to")
    return op if callable(op) else None


def grid():
    return [
        {"src_shape": list(src), "target_shape": list(target)}
        for src, target in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    src_shape = tuple(binding["src_shape"])
    x = torch.randn(src_shape, dtype=dtype, device=device)
    mask = torch.rand(src_shape, device=device) < _DENSITY
    inp = (x * mask).to_sparse().coalesce()
    return (inp, tuple(binding["target_shape"])), {}


def key_shape(binding):
    src = "x".join(str(d) for d in binding["src_shape"])
    target = "x".join(str(d) for d in binding["target_shape"])
    return f"src{src}_to{target}"


def config(binding, dtype):
    src = list(binding["src_shape"])
    target = list(binding["target_shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {
                    "shape": src,
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
        "dims": {"src_shape": src, "target_shape": target},
    }
