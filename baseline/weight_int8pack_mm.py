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

"""NVIDIA native baseline for `weight_int8pack_mm`.

Mirrors benchmark/test_weight_int8pack_mm.py (WeightInt8packMMBenchmark, a
base.Benchmark subclass):
    op_name  = "weight_int8pack_mm"
    torch_op = weight_int8pack_mm_torch (the module-local reference below)
    dtypes   = [torch.float16, torch.bfloat16]

The class overrides set_shapes with WEIGHT_INT8PACK_MM_SHAPES -- read as
(M, N, K) -- so core_shapes.yaml is not consulted. The list sweeps M over
1/4/16/32/64/128 against Llama-shaped N x K pairs.

CAUTION: the benchmark's reference is not an ATen int8 GEMM. It dequantizes the
int8 weight to the activation dtype, runs a dense torch.matmul and scales the
result -- so the baseline measures a dequantize + dense matmul, not a packed
int8 kernel. That composition is reproduced here. See Q&A.md.
"""

import torch

OP_NAME = "weight_int8pack_mm"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.matmul is unavailable on this torch build"

# test_weight_int8pack_mm.py::WEIGHT_INT8PACK_MM_SHAPES -- (M, N, K)
_SHAPES = [
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    (1, 8192, 8192),
    (1, 8192, 28672),
    (1, 28672, 8192),
    (4, 4096, 4096),
    (4, 4096, 11008),
    (4, 11008, 4096),
    (16, 4096, 4096),
    (16, 4096, 11008),
    (16, 11008, 4096),
    (32, 4096, 4096),
    (32, 4096, 11008),
    (32, 11008, 4096),
    (64, 4096, 4096),
    (64, 4096, 11008),
    (64, 11008, 4096),
    (128, 4096, 4096),
    (128, 4096, 11008),
    (128, 11008, 4096),
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


def _weight_int8pack_mm_torch(a, b, scales):
    """benchmark::weight_int8pack_mm_torch -- dequantize, matmul, scale."""
    b_fp = b.to(a.dtype)
    result = torch.matmul(a, b_fp.T)
    result = result * scales.unsqueeze(0)
    return result


def native():
    if _resolve("torch.matmul") is None:
        return None
    return _weight_int8pack_mm_torch


def grid():
    return [{"m": m, "n": n, "k": k} for m, n, k in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n, k = binding["m"], binding["n"], binding["k"]
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randint(-128, 127, (n, k), dtype=torch.int8, device=device)
    scales = torch.randn((n,), dtype=dtype, device=device)
    return (a, b, scales), {}


def key_shape(binding):
    return [binding["m"], binding["n"], binding["k"]]


def config(binding, dtype):
    m, n, k = binding["m"], binding["n"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [m, k], "dtype": dt},
                {"shape": [n, k], "dtype": "torch.int8"},
                {"shape": [n], "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [m, n], "dtype": dt}},
        "dims": {"M": m, "N": n, "K": k},
    }
