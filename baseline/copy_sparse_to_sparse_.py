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

"""NVIDIA native baseline for `copy_sparse_to_sparse_`.

Mirrors benchmark/test_copy_sparse_to_sparse_.py (CopySparseToSparseBenchmark,
a base.Benchmark subclass):
    op_name    = "copy_sparse_to_sparse_"
    torch_op   = torch.ops.aten.copy_sparse_to_sparse_.default
    dtypes     = consts.FLOAT_DTYPES
    is_inplace = True

The class overrides set_shapes with its own 1-D size list, so core_shapes.yaml
is not consulted. Each size builds a coalesced COO source and a preallocated
COO destination with matching nnz (indices are arange, so nnz == size).
"""

import torch

OP_NAME = "copy_sparse_to_sparse_"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.copy_sparse_to_sparse_.default is unavailable on this build"
)

# CopySparseToSparseBenchmark.set_shapes -- preallocated COO destinations
_SHAPES = [
    (64,),
    (4096,),
    (65536,),
    (262144,),
    (1048576,),
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
    op = _resolve("torch.ops.aten.copy_sparse_to_sparse_.default")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    (size,) = binding["shape"]
    idx = torch.arange(size, device=device).unsqueeze(0)
    values = torch.randn(size, dtype=dtype, device=device)
    src = torch.sparse_coo_tensor(idx, values, (size,), is_coalesced=True)
    dst = torch.sparse_coo_tensor(idx.clone(), torch.empty_like(values), (size,))
    return (dst, src), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    (size,) = binding["shape"]
    dt = str(dtype)
    sparse = {
        "shape": [size],
        "dtype": dt,
        "layout": "torch.sparse_coo",
        "nnz": size,
    }
    return {
        "inputs": {
            "args": [
                dict(sparse, note="written in place"),
                dict(sparse, is_coalesced=True),
            ],
            "kwargs": {},
        },
        "outputs": {"self": dict(sparse)},
        "dims": {"size": size, "nnz": size},
    }
