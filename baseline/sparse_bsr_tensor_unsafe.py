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

"""NVIDIA native baseline for `sparse_bsr_tensor_unsafe`.

Mirrors benchmark/test_sparse_bsr_tensor_unsafe.py
(SparseBsrTensorUnsafeBenchmark, a base.Benchmark subclass):
    op_name  = "sparse_bsr_tensor_unsafe"
    torch_op = native_bsr (defined inside the test function)
    dtypes   = [torch.float32]

The class overrides set_shapes with BSR_UNSAFE_NNZ -- a list of nnz counts, not
tensor shapes -- so core_shapes.yaml is not consulted. The logical size is
always (4, 4, 2) with a fixed crow of [0, 1, 2, 3, 4] and (2, 2) value blocks;
only the block-index and value lengths vary with nnz, which is deliberately
inconsistent with crow -- this is the *unsafe* constructor, which skips
validation.

The benchmark wraps the operator to pass device=values.device explicitly: the
native unsafe constructor fails its internal values/instance device check when
the keyword is omitted.
"""

import torch

OP_NAME = "sparse_bsr_tensor_unsafe"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._sparse_bsr_tensor_unsafe is unavailable on this build"
)

# test_sparse_bsr_tensor_unsafe.py::BSR_UNSAFE_NNZ
_NNZ_VALUES = [1, 64, 4096]

_CROW = [0, 1, 2, 3, 4]
_BLOCK = (2, 2)
_SIZE = (4, 4, 2)


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


def _native_bsr(crow, col, values, size):
    """benchmark::test_sparse_bsr_tensor_unsafe::native_bsr."""
    return torch.ops.aten._sparse_bsr_tensor_unsafe(
        crow, col, values, size, device=values.device
    )


def native():
    if _resolve("torch.ops.aten._sparse_bsr_tensor_unsafe") is None:
        return None
    return _native_bsr


def grid():
    return [{"nnz": nnz} for nnz in _NNZ_VALUES]


def build_inputs(binding, dtype, device):
    nnz = binding["nnz"]
    crow = torch.tensor(_CROW, dtype=torch.int64, device=device)
    col = torch.randint(0, 2, (nnz,), device=device).to(torch.int64)
    values = torch.randn((nnz, *_BLOCK), dtype=dtype, device=device)
    return (crow, col, values, _SIZE), {}


def key_shape(binding):
    return f"nnz{binding['nnz']}"


def config(binding, dtype):
    nnz = binding["nnz"]
    dt = str(dtype)
    i64 = "torch.int64"
    return {
        "inputs": {
            "args": [
                {"shape": [len(_CROW)], "dtype": i64, "note": "crow"},
                {"shape": [nnz], "dtype": i64, "note": "block col"},
                {"shape": [nnz, *_BLOCK], "dtype": dt},
                list(_SIZE),
            ],
            "kwargs": {},
        },
        "outputs": {
            "out": {
                "shape": list(_SIZE),
                "dtype": dt,
                "layout": "torch.sparse_bsr",
            }
        },
        "dims": {"nnz": nnz, "block": list(_BLOCK), "size": list(_SIZE)},
    }
