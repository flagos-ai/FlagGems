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

"""NVIDIA native baseline for `pdist_backward`.

Mirrors benchmark/test_pdist_backward.py (PdistBackwardBenchmark, a
base.Benchmark subclass):
    op_name  = "pdist_backward"
    torch_op = torch.ops.aten._pdist_backward
    dtypes   = [torch.float32]

The class overrides set_shapes with PDIST_BACKWARD_SHAPES -- read as (n, m) --
so core_shapes.yaml is not consulted.

The forward pdist result is computed in build_inputs because the backward takes
it as an argument; the gradient is ones of length n * (n - 1) // 2 and p is
2.0.

float32 only: the benchmark pins it for numerical stability.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "pdist_backward"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._pdist_backward is unavailable on this torch build"
)

# test_pdist_backward.py::PDIST_BACKWARD_SHAPES -- (n, m)
_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]

_P = 2.0


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
    op = _resolve("torch.ops.aten._pdist_backward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    n, _ = binding["shape"]
    n_pairs = n * (n - 1) // 2
    x = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    pdist_out = torch.pdist(x, p=_P)
    grad = torch.ones(n_pairs, dtype=dtype, device=device)
    return (grad, x, _P, pdist_out), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    n = shape[0]
    n_pairs = n * (n - 1) // 2
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [n_pairs], "dtype": dt, "note": "grad"},
                {"shape": shape, "dtype": dt},
                {"scalar": _P},
                {"shape": [n_pairs], "dtype": dt, "note": "forward pdist"},
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {"N": n, "M": shape[1], "n_pairs": n_pairs, "p": _P},
    }
