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

"""NVIDIA native baseline for `lu_unpack_backward`.

Mirrors benchmark/test_lu_unpack.py (LuUnpackBackwardBenchmark, a
GenericBenchmark2DOnly subclass, is_backward=True):
    op_name  = "lu_unpack" + "_backward" (appended by base.Benchmark)
    torch_op = torch.ops.aten.lu_unpack
    dtypes   = [torch.float32, torch.float64]
    shapes   = yaml:GenericBenchmark2DOnly (benchmark/core_shapes.yaml),
               then filtered by Benchmark.MAX_BACKWARD_ELEMENTS (2**27)

The class overrides _benchmark_callable rather than relying on the generic
is_backward path: it runs `op(LU_data, LU_pivots)` once outside the timed
closure and then times

    torch.autograd.grad((L, U), (LU_data,), (grad_L, grad_U), retain_graph=True)

so the measured latency is the packed-LU gradient path -- two grad_outputs, not
one. build_inputs() reproduces that preparation and `_backward_lu_unpack` is
the timed call.

LU and pivots come from a real torch.linalg.lu_factor, which supports float32
and float64 only -- hence the dtype list.
"""

import torch

OP_NAME = "lu_unpack_backward"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.lu_unpack is unavailable on this torch build"
)

# yaml:GenericBenchmark2DOnly
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
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


def _backward_lu_unpack(low, upper, lu_data, grad_low, grad_upper):
    """The timed region of LuUnpackBackwardBenchmark._benchmark_callable."""
    return torch.autograd.grad(
        (low, upper), (lu_data,), (grad_low, grad_upper), retain_graph=True
    )


def native():
    if _resolve("torch.ops.aten.lu_unpack") is None:
        return None
    return _backward_lu_unpack


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["shape"]
    a = torch.randn(m, n, dtype=dtype, device=device)
    lu, pivots = torch.linalg.lu_factor(a)
    lu = lu.detach().requires_grad_(True)
    _, low, upper = torch.ops.aten.lu_unpack(lu, pivots)
    grad_low = torch.randn_like(low)
    grad_upper = torch.randn_like(upper)
    return (low, upper, lu, grad_low, grad_upper), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    m, n = binding["shape"]
    k = min(m, n)
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [m, n], "dtype": dt, "requires_grad": True},
                {"shape": [k], "dtype": "torch.int32"},
            ],
            "kwargs": {},
        },
        "outputs": {"grad_LU": {"shape": [m, n], "dtype": dt}},
        "dims": {"M": m, "N": n, "K": k},
    }
