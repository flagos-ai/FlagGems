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

"""NVIDIA native baseline for `permute_copy`.

Mirrors benchmark/test_permute_copy.py (PermuteCopyBenchmark, a base.Benchmark
subclass):
    op_name  = "permute_copy"
    torch_op = torch.permute_copy
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with PERMUTE_COPY_SHAPES -- (shape, dims)
pairs -- so core_shapes.yaml is not consulted. The second entry's identity
permutation (0, 1) is deliberate: it measures the copy with no reordering.
"""

import torch

OP_NAME = "permute_copy"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.permute_copy is unavailable on this torch build"

# test_permute_copy.py::PERMUTE_COPY_SHAPES -- (shape, dims)
_SHAPES = [
    ((64, 128), (1, 0)),
    ((128, 256), (0, 1)),
    ((32, 64, 128), (2, 0, 1)),
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
    op = _resolve("torch.permute_copy")
    return op if callable(op) else None


def grid():
    return [{"shape": list(shape), "dims": list(dims)} for shape, dims in _SHAPES]


def build_inputs(binding, dtype, device):
    x = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (x, tuple(binding["dims"])), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    dims = "".join(str(d) for d in binding["dims"])
    return f"{shape}_perm{dims}"


def config(binding, dtype):
    shape = list(binding["shape"])
    dims = list(binding["dims"])
    dt = str(dtype)
    out_shape = [shape[d] for d in dims]
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                dims,
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {"shape": shape, "permutation": dims},
    }
