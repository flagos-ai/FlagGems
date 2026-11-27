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

"""NVIDIA native baseline for `addmm_dtype_out`.

Mirrors benchmark/test_addmm.py (base.BlasBenchmark):
    op_name  = "addmm_dtype_out"
    torch_op = lambda bias, mat1, mat2, out_dtype, out:
                   torch.ops.aten.addmm.dtype_out(
                       bias, mat1, mat2, out_dtype, beta=1.0, alpha=1.0, out=out
                   )
    dtypes   = consts.FP16_BF16_DTYPES  ([float16, bfloat16])
    shapes   = yaml:BlasBenchmark (benchmark/core_shapes.yaml)

The B dimension is unused by this 2-D operator: it is carried through the
shared "B, M, N, K" BLAS shape table. benchmark/gemm_shapes.py::blas_cases only
expands the column-major layout for the names in GEMM_OPS (mm, addmm, bmm,
baddbmm, router_gemm) or at --level comprehensive, so this operator runs with
b_column_major=False only and mat2 stays row-major [k, n].

The benchmark passes `out` positionally into its lambda and beta/alpha
explicitly as 1.0; `_native_addmm_dtype_out` is the kunlunxin-only reference
and is not used on NVIDIA. The wrapper below keeps that exact call shape, so
the measured kernel sees the same bound arguments.
"""

import torch

OP_NAME = "addmm_dtype_out"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = (4,)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.addmm.dtype_out is unavailable on this torch build"
)

# yaml:BlasBenchmark -- shape_desc "B, M, N, K"
_SHAPES = [
    (2, 384, 384, 384),
    (2, 4096, 4096, 4096),
    (16, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (16, 4096, 4096, 4096),
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


def _addmm_dtype_out(bias, mat1, mat2, out_dtype, out):
    """The benchmark's torch_op lambda, with beta/alpha pinned to 1.0."""
    return torch.ops.aten.addmm.dtype_out(
        bias, mat1, mat2, out_dtype, beta=1.0, alpha=1.0, out=out
    )


def native():
    if _resolve("torch.ops.aten.addmm.dtype_out") is None:
        return None
    return _addmm_dtype_out


def grid():
    return [
        {"b": b, "m": m, "n": n, "k": k, "b_column_major": False}
        for b, m, n, k in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    m, n, k = binding["m"], binding["n"], binding["k"]
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=torch.float32, device=device)
    out = torch.empty([m, n], dtype=torch.float32, device=device)
    inp2 = torch.randn([k, n], dtype=dtype, device=device)
    return (bias, inp1, inp2, torch.float32, out), {}


def key_shape(binding):
    b, m, n, k = binding["b"], binding["m"], binding["n"], binding["k"]
    return f"b{b}_m{m}_n{n}_k{k}"


def config(binding, dtype):
    m, n, k = binding["m"], binding["n"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [m, n], "dtype": "torch.float32"},
                {"shape": [m, k], "dtype": dt},
                {"shape": [k, n], "dtype": dt},
                "torch.float32",
                {"shape": [m, n], "dtype": "torch.float32"},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [m, n], "dtype": "torch.float32"}},
        "dims": {"B": binding["b"], "M": m, "N": n, "K": k},
    }
