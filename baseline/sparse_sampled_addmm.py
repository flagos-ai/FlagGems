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

"""NVIDIA baseline for sparse_sampled_addmm (torch native).

Self-contained module; depends only on torch. ``native()`` returns the torch
implementation used as the NVIDIA baseline: ``torch.sparse.sampled_addmm`` on
fp32/fp64, with the same fp32 upcast the benchmark applies for fp16/bf16
(torch has no Half/BFloat16 kernel). Shapes, dtypes and the sparsity are copied
from ``benchmark/test_sparse_sampled_addmm.py`` and
``benchmark/core_shapes.yaml``.
"""

import torch

OP_NAME = "sparse_sampled_addmm"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/core_shapes.yaml::BlasBenchmark (B, M, N, K)
_CASES = [
    (2, 384, 384, 384),
    (2, 4096, 4096, 4096),
    (16, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (16, 4096, 4096, 4096),
]
_SPARSITY = 0.9


def _make_sparse_input(b, m, n, dtype, device):
    # Mirrors benchmark/test_sparse_sampled_addmm.py::_make_sparse_input.
    dense = torch.randn((b, m, n), dtype=dtype, device=device)
    dense = torch.where(dense == 0, 1.0, dense)
    mask = torch.rand((m, n), device=device) > _SPARSITY
    mask = mask.expand(b, m, n)
    dense = (dense * mask).contiguous()
    return dense.to_sparse_csr()


def _sampled_addmm(input, mat1, mat2, *, alpha=1.0, beta=1.0, out=None):
    if mat1.dtype not in (torch.float16, torch.bfloat16):
        return torch.sparse.sampled_addmm(
            input, mat1, mat2, alpha=alpha, beta=beta, out=out
        )
    input32 = torch.sparse_csr_tensor(
        input.crow_indices(),
        input.col_indices(),
        input.values().to(torch.float32),
        size=input.shape,
        device=input.device,
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
    return _sampled_addmm


def grid():
    return [{"shape": case} for case in _CASES]


def build_inputs(binding, dtype, device):
    b, m, n, k = binding["shape"]
    mat1 = torch.randn((b, m, k), dtype=dtype, device=device)
    mat2 = torch.randn((b, k, n), dtype=dtype, device=device)
    sparse_input = _make_sparse_input(b, m, n, dtype, device)
    return (sparse_input, mat1, mat2), {}


def key_shape(binding):
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    b, m, n, k = binding["shape"]
    return {
        "inputs": {
            "input": {"shape": [b, m, n], "dtype": str(dtype)},
            "mat1": {"shape": [b, m, k], "dtype": str(dtype)},
            "mat2": {"shape": [b, k, n], "dtype": str(dtype)},
        },
        "outputs": {
            "out": {"shape": [b, m, n], "dtype": str(dtype)},
        },
        "dims": {"b": b, "m": m, "n": n, "k": k, "sparsity": _SPARSITY},
    }
