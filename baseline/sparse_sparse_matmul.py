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

"""NVIDIA native baseline for `sparse_sparse_matmul`.

Mirrors benchmark/test_sparse_sparse_matmul.py (SparseSparseMatmulBenchmark, a
BlasBenchmark subclass):
    op_name  = "sparse_sparse_matmul"
    torch_op = _torch_sparse_sparse_matmul, a pass-through to
               torch.ops.aten._sparse_sparse_matmul
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:BlasBenchmark (benchmark/core_shapes.yaml),
               shape_desc "B, M, N, K"

The class's get_input_iter ignores the B dimension and passes the shape's index
as the RNG seed; set_more_shapes returns [], so the grid is the five yaml
shapes. Both operands are COO tensors at 5% density, built with a seeded CPU
generator (seed i for A, i + 1 for B) -- so the grid carries that index.
"""

import torch

OP_NAME = "sparse_sparse_matmul"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._sparse_sparse_matmul is unavailable on this torch build"
)

# yaml:BlasBenchmark -- shape_desc "B, M, N, K"
_SHAPES = [
    (2, 384, 384, 384),
    (2, 4096, 4096, 4096),
    (16, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (16, 4096, 4096, 4096),
]

_DENSITY = 0.05


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


def _make_sparse_coo(shape, dtype, device, density=_DENSITY, seed=0):
    """benchmark::_make_sparse_coo -- seeded COO pattern at `density`."""
    m, n = shape[-2:]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    nnz = max(1, int(m * n * density))
    flat = torch.randperm(m * n, generator=generator)[:nnz]
    rows = (flat // n).to(torch.int64).to(device)
    cols = (flat % n).to(torch.int64).to(device)
    values = (
        torch.randn(nnz, generator=generator, dtype=torch.float32).to(dtype).to(device)
    )
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), values, (m, n), device=device
    ).coalesce()


def _torch_sparse_sparse_matmul(a, b):
    """benchmark::_torch_sparse_sparse_matmul."""
    return torch.ops.aten._sparse_sparse_matmul(a, b)


def native():
    if _resolve("torch.ops.aten._sparse_sparse_matmul") is None:
        return None
    return _torch_sparse_sparse_matmul


def grid():
    return [
        {"b": b, "m": m, "n": n, "k": k, "seed": i}
        for i, (b, m, n, k) in enumerate(_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    m, n, k = binding["m"], binding["n"], binding["k"]
    seed = binding["seed"]
    a = _make_sparse_coo((m, k), dtype, device, seed=seed)
    b = _make_sparse_coo((k, n), dtype, device, seed=seed + 1)
    return (a, b), {}


def key_shape(binding):
    return f"b{binding['b']}_m{binding['m']}" f"_n{binding['n']}_k{binding['k']}"


def config(binding, dtype):
    m, n, k = binding["m"], binding["n"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {
                    "shape": [m, k],
                    "dtype": dt,
                    "layout": "torch.sparse_coo",
                    "density": _DENSITY,
                },
                {
                    "shape": [k, n],
                    "dtype": dt,
                    "layout": "torch.sparse_coo",
                    "density": _DENSITY,
                },
            ],
            "kwargs": {},
        },
        "outputs": {
            "out": {
                "shape": [m, n],
                "dtype": dt,
                "layout": "torch.sparse_coo",
            }
        },
        "dims": {
            "B": binding["b"],
            "M": m,
            "N": n,
            "K": k,
            "density": _DENSITY,
            "seed": binding["seed"],
        },
    }
