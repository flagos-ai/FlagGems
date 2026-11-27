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

"""NVIDIA native baseline for `sparse_sampled_addmm_out`.

Mirrors benchmark/test_sparse_sampled_addmm.py (SparseSampledAddmmOutBenchmark, a
BlasBenchmark subclass):
    op_name  = "sparse_sampled_addmm_out"
    torch_op = _torch_sampled_addmm (the module-local reference below)
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:sparse_sampled_addmm_out (benchmark/core_shapes.yaml),
               shape_desc "B, M, N, K"

The class's get_input_iter passes b_column_major=False and set_more_shapes
returns [], so the grid is the twelve yaml shapes.

`_torch_sampled_addmm` has three branches. The dense-recompute fallback applies
only where the native kernel is missing (not NVIDIA/metax/hygon), so on NVIDIA
the reference is torch.sparse.sampled_addmm -- directly for fp32, and through
an fp32 upcast of the CSR values and both dense operands for fp16/bf16, since
the native kernel has no low-precision path. Both NVIDIA branches are
reproduced here.

CAUTION on the low-precision rows: the upcast runs inside the timed region, so
fp16/bf16 speedups read as "kernel vs upcast-aten" and only the fp32 row is
like-for-like. See Q&A.md.

The sparse operand is a CSR tensor at 90% sparsity whose dense values have
their exact zeros replaced by 1.0 first, so the stored pattern is exactly the
mask. The `_dense_twin` / `_dense_mask` / `_dense_flat_idx` attributes the
benchmark attaches are only read by the non-NVIDIA fallback and are not
needed here.

The .out variant writes into a clone of the same CSR tensor.
"""

import torch

OP_NAME = "sparse_sampled_addmm_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.sparse.sampled_addmm is unavailable on this torch build"
)

# yaml:sparse_sampled_addmm_out -- shape_desc "B, M, N, K"
_SHAPES = [
    (2, 128, 128, 128),
    (4, 256, 256, 256),
    (8, 512, 512, 512),
    (16, 512, 512, 512),
    (4, 1024, 1024, 1024),
    (16, 1024, 1024, 1024),
    (8, 2048, 2048, 64),
    (2, 2048, 2048, 128),
    (4, 4096, 4096, 128),
    (1, 4096, 4096, 64),
    (2, 4096, 1024, 512),
    (2, 1024, 1000, 384),
]

_SPARSITY = 0.9
_ALPHA = 1.0
_BETA = 1.0


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


def _make_sparse_input(shape, dtype, device, sparsity=_SPARSITY):
    """benchmark::_make_sparse_input -- CSR whose pattern is exactly the mask."""
    m, n = shape[-2:]
    dense = torch.randn(shape, dtype=dtype, device=device)
    dense = torch.where(dense == 0, 1.0, dense)
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


def _torch_sampled_addmm(inp, mat1, mat2, *, alpha=_ALPHA, beta=_BETA, out=None):
    """benchmark::_torch_sampled_addmm, NVIDIA branches only."""
    if mat1.dtype not in (torch.float16, torch.bfloat16):
        return torch.sparse.sampled_addmm(
            inp, mat1, mat2, alpha=alpha, beta=beta, out=out
        )
    input32 = torch.sparse_csr_tensor(
        inp.crow_indices(),
        inp.col_indices(),
        inp.values().to(torch.float32),
        size=inp.shape,
        device=inp.device,
    )
    result = torch.sparse.sampled_addmm(
        input32,
        mat1.to(torch.float32),
        mat2.to(torch.float32),
        alpha=alpha,
        beta=beta,
    )
    new_values = result.values().to(mat1.dtype)
    if out is not None:
        out.values().copy_(new_values)
        return out
    return torch.sparse_csr_tensor(
        result.crow_indices(),
        result.col_indices(),
        new_values,
        size=result.shape,
        device=result.device,
    )


def native():
    if _resolve("torch.sparse.sampled_addmm") is None:
        return None
    return _torch_sampled_addmm


def grid():
    return [{"b": b, "m": m, "n": n, "k": k} for b, m, n, k in _SHAPES]


def build_inputs(binding, dtype, device):
    b, m, n, k = binding["b"], binding["m"], binding["n"], binding["k"]
    mat1 = torch.randn([b, m, k], dtype=dtype, device=device)
    mat2 = torch.randn([b, k, n], dtype=dtype, device=device)
    sparse_input = _make_sparse_input([b, m, n], dtype, device, _SPARSITY)
    # The .out buffer is a clone of the same CSR tensor -- identical pattern,
    # independent storage, as the benchmark's _input_fn_out builds it.
    out = torch.sparse_csr_tensor(
        sparse_input.crow_indices().clone(),
        sparse_input.col_indices().clone(),
        sparse_input.values().clone(),
        size=sparse_input.shape,
        dtype=sparse_input.dtype,
        device=sparse_input.device,
    )
    return (sparse_input, mat1, mat2), {"out": out}


def key_shape(binding):
    return f"b{binding['b']}_m{binding['m']}" f"_n{binding['n']}_k{binding['k']}"


def config(binding, dtype):
    b, m, n, k = binding["b"], binding["m"], binding["n"], binding["k"]
    dt = str(dtype)
    sparse = {
        "shape": [b, m, n],
        "dtype": dt,
        "layout": "torch.sparse_csr",
        "sparsity": _SPARSITY,
    }
    return {
        "inputs": {
            "args": [
                dict(sparse),
                {"shape": [b, m, k], "dtype": dt},
                {"shape": [b, k, n], "dtype": dt},
            ],
            "kwargs": {"out": dict(sparse)},
        },
        "outputs": {"out": dict(sparse)},
        "dims": {"B": b, "M": m, "N": n, "K": k, "sparsity": _SPARSITY},
    }
