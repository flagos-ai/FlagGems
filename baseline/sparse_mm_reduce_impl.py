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

"""NVIDIA native baseline for `sparse_mm_reduce_impl`.

Mirrors benchmark/test_sparse_mm_reduce_impl.py (SparseMmReduceImplBenchmark,
a BlasBenchmark subclass):
    op_name  = "sparse_mm_reduce_impl"
    torch_op = _TorchSparseMmReduceImpl() (a callable instance)
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:sparse_mm_reduce_impl (benchmark/core_shapes.yaml),
               shape_desc "B, M, N, K"

The class's get_input_iter ignores the B dimension and set_more_shapes returns
[], so the grid is the six yaml shapes. mat1 is a CSR tensor at 90% sparsity.

CAUTION: `aten::_sparse_mm_reduce_impl` has no CUDA registration. The
benchmark's baseline moves both operands to the host, runs the CPU kernel and
moves the result back -- and for fp16/bf16 additionally upcasts to fp32,
because the CPU sparse-reduce kernel has no low-precision path. So every row
here is a CPU measurement plus transfers, and NCU will find no device kernel.
That wrapper is reproduced verbatim. See Q&A.md.

`reduce` defaults to "sum".
"""

import torch

OP_NAME = "sparse_mm_reduce_impl"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._sparse_mm_reduce_impl is unavailable on this torch build"
)

# yaml:sparse_mm_reduce_impl -- shape_desc "B, M, N, K"
_SHAPES = [
    (2, 512, 512, 512),
    (8, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (4, 4096, 4096, 4096),
    (2, 4096, 1024, 4096),
    (8, 4096, 256, 4096),
]

_SPARSITY = 0.9
_REDUCE = "sum"


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


def _torch_sparse_mm_reduce_impl(self_sparse, other, reduce=_REDUCE):
    """benchmark::_TorchSparseMmReduceImpl.__call__ -- CPU kernel, fp32 upcast."""
    move = other.device.type == "cuda"
    if move:
        self_sparse = self_sparse.cpu()
        other = other.cpu()
    if self_sparse.dtype in (torch.float16, torch.bfloat16):
        self_sparse = self_sparse.to(torch.float32).to_sparse_csr()
        other = other.to(torch.float32)
    out, _ = torch.ops.aten._sparse_mm_reduce_impl(self_sparse, other, reduce)
    if move:
        out = out.to(other.device)
    return out


def native():
    if _resolve("torch.ops.aten._sparse_mm_reduce_impl") is None:
        return None
    return _torch_sparse_mm_reduce_impl


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
            "reduce": _REDUCE,
        },
    }
