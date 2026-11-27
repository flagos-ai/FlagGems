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

"""NVIDIA native baseline for `sparse_mm`.

Mirrors benchmark/test_sparse_mm.py (SparseMmBenchmark, a BlasBenchmark
subclass):
    op_name  = "sparse_mm"
    torch_op = _torch_sparse_mm (the module-local reference below)
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:sparse_mm (benchmark/core_shapes.yaml),
               shape_desc "B, M, N, K"

The class's get_input_iter ignores the B dimension and always passes
b_column_major=False, and set_more_shapes returns [], so the grid is the six
yaml shapes. mat1 is a CSR tensor at 90% sparsity.

CAUTION on the low-precision rows: `aten::_sparse_mm` has no fp16/bf16 CUDA
kernel, so the reference upcasts to fp32. The benchmark memoises that cast
per input identity (`_upcast_inputs`) so do_bench's repeated calls pay it once
rather than every iteration. The cache is reproduced here for the same reason:
without it the baseline would pay a cast the candidate does not. Read the
fp16/bf16 speedups as "kernel vs upcast-aten" and the fp32 row as the
like-for-like comparison. See Q&A.md.
"""

import weakref

import torch

OP_NAME = "sparse_mm"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._sparse_mm is unavailable on this torch build"
)

# yaml:sparse_mm -- shape_desc "B, M, N, K"
_SHAPES = [
    (2, 512, 512, 512),
    (8, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (4, 4096, 4096, 4096),
    (2, 4096, 1024, 4096),
    (8, 4096, 256, 4096),
]

_SPARSITY = 0.9
_UPCAST_CACHE = {}


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


def _make_sparse_csr(shape, dtype, device, sparsity=_SPARSITY):
    """benchmark::_make_sparse_csr -- masked dense, then CSR."""
    m, n = shape[-2:]
    dense = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand((m, n), device=device) > sparsity
    mask = mask.expand(shape)
    dense = (dense * mask).contiguous()
    if torch.device(device).type == "cuda":
        return dense.to_sparse_csr()
    csr_cpu = dense.cpu().to_sparse_csr()
    return torch.sparse_csr_tensor(
        csr_cpu.crow_indices().to(device),
        csr_cpu.col_indices().to(device),
        csr_cpu.values().to(device),
        size=csr_cpu.shape,
        dtype=csr_cpu.dtype,
        device=device,
    )


def _upcast_inputs(mat1, mat2):
    """benchmark::_upcast_inputs -- fp32 copies cached by input identity."""
    key = (id(mat1), id(mat2))
    entry = _UPCAST_CACHE.get(key)
    if (
        entry is None
        or entry[0]() is not mat1
        or entry[1]() is not mat2
        or entry[2].device != mat1.device
    ):
        entry = (weakref.ref(mat1), weakref.ref(mat2), mat1.float(), mat2.float())
        _UPCAST_CACHE[key] = entry
    return entry[2], entry[3]


def _torch_sparse_mm(mat1, mat2):
    """benchmark::_torch_sparse_mm -- fp32 upcast for the half dtypes."""
    if mat1.dtype in (torch.float16, torch.bfloat16):
        upcast_sparse, upcast_dense = _upcast_inputs(mat1, mat2)
        return torch.ops.aten._sparse_mm(upcast_sparse, upcast_dense).to(mat1.dtype)
    return torch.ops.aten._sparse_mm(mat1, mat2)


def native():
    if _resolve("torch.ops.aten._sparse_mm") is None:
        return None
    return _torch_sparse_mm


def grid():
    return [{"b": b, "m": m, "n": n, "k": k} for b, m, n, k in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n, k = binding["m"], binding["n"], binding["k"]
    mat1 = _make_sparse_csr((m, k), dtype, device, _SPARSITY)
    mat2 = torch.randn([k, n], dtype=dtype, device=device)
    return (mat1, mat2), {}


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
                    "layout": "torch.sparse_csr",
                    "sparsity": _SPARSITY,
                },
                {"shape": [k, n], "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [m, n], "dtype": dt}},
        "dims": {
            "B": binding["b"],
            "M": m,
            "N": n,
            "K": k,
            "sparsity": _SPARSITY,
        },
    }
