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

"""NVIDIA native baseline for `validate_sparse_csc_tensor_args`.

Mirrors benchmark/test_validate_sparse_csc_tensor_args.py (SparseValidationBenchmark, a
base.Benchmark subclass):
    op_name  = "validate_sparse_csc_tensor_args"
    torch_op = torch.ops.aten._validate_sparse_csc_tensor_args.default
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own (cdim, dim, batch, dense)
descriptors -- short and long compressed axes plus one batched case -- so
core_shapes.yaml is not consulted.

`_make_inputs` builds a structurally valid CSC triple: the compressed
pointers are arange(cdim + 1) * per_segment with per_segment = min(dim, 2), the
plain indices are arange(per_segment) tiled cdim times, and the values are
zeros. Index tensors are int64 and `strided_batch` is False.

This operator validates metadata rather than computing on it, so the measured
work is shape and index checking.
"""

import torch

OP_NAME = "validate_sparse_csc_tensor_args"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.ops.aten._validate_sparse_csc_tensor_args.default is unavailable on this build"

# SparseValidationBenchmark.set_shapes -- (cdim, dim, batch, dense)
_SHAPES = [
    (3, 5, (), ()),
    (256, 256, (), ()),
    (4096, 4096, (), ()),
    (64, 64, (8,), ()),
]

_ROW_MAJOR = False
_BLOCKED = False
_BLOCK_SHAPE = (1, 1)


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


def _make_inputs(shape, index_dtype, dtype, device):
    """benchmark::_make_inputs for this layout, strided_batch=False."""
    cdim, dim, batch, dense = shape
    batch = tuple(batch)
    dense = tuple(dense)
    per_segment = min(dim, 2)
    nnz = cdim * per_segment
    comp = torch.arange(cdim + 1, dtype=index_dtype, device=device) * per_segment
    plain = torch.arange(per_segment, dtype=index_dtype, device=device).repeat(cdim)
    comp = comp.expand(*batch, cdim + 1).clone()
    plain = plain.expand(*batch, nnz).clone()
    br, bc = _BLOCK_SHAPE if _BLOCKED else (1, 1)
    rows, cols = (cdim, dim) if _ROW_MAJOR else (dim, cdim)
    block_shape = (br, bc) if _BLOCKED else ()
    values = torch.zeros(
        (*batch, nnz, *block_shape, *dense), dtype=dtype, device=device
    )
    size = (*batch, rows * br, cols * bc, *dense)
    return comp, plain, values, size


def native():
    op = _resolve("torch.ops.aten._validate_sparse_csc_tensor_args.default")
    return op if callable(op) else None


def grid():
    return [
        {
            "cdim": cdim,
            "dim": dim,
            "batch": list(batch),
            "dense": list(dense),
        }
        for cdim, dim, batch, dense in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    shape = (
        binding["cdim"],
        binding["dim"],
        tuple(binding["batch"]),
        tuple(binding["dense"]),
    )
    comp, plain, values, size = _make_inputs(shape, torch.int64, dtype, device)
    return (comp, plain, values, size), {}


def key_shape(binding):
    batch = "x".join(str(d) for d in binding["batch"]) or "nobatch"
    return f"cdim{binding['cdim']}_dim{binding['dim']}_batch{batch}"


def config(binding, dtype):
    cdim, dim = binding["cdim"], binding["dim"]
    batch = list(binding["batch"])
    dense = list(binding["dense"])
    per_segment = min(dim, 2)
    nnz = cdim * per_segment
    br, bc = _BLOCK_SHAPE if _BLOCKED else (1, 1)
    rows, cols = (cdim, dim) if _ROW_MAJOR else (dim, cdim)
    block_shape = [br, bc] if _BLOCKED else []
    dt = str(dtype)
    i64 = "torch.int64"
    return {
        "inputs": {
            "args": [
                {"shape": batch + [cdim + 1], "dtype": i64},
                {"shape": batch + [nnz], "dtype": i64},
                {"shape": batch + [nnz] + block_shape + dense, "dtype": dt},
                batch + [rows * br, cols * bc] + dense,
            ],
            "kwargs": {},
        },
        "outputs": {"out": None},
        "dims": {
            "cdim": cdim,
            "dim": dim,
            "batch": batch,
            "dense": dense,
            "nnz": nnz,
        },
    }
