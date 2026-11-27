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

"""NVIDIA native baseline for `logcumsumexp_backward`.

Mirrors benchmark/test_logcumsumexp.py (base.GenericBenchmark2DOnly,
is_backward=True):
    op_name  = "logcumsumexp" + "_backward" (appended by base.Benchmark)
    torch_op = torch.logcumsumexp
    dtypes   = [torch.float32, torch.bfloat16]
    shapes   = yaml:GenericBenchmark2DOnly (benchmark/core_shapes.yaml),
               then filtered by Benchmark.MAX_BACKWARD_ELEMENTS (2**27)

float16 is deliberately absent from the dtype list: torch raises
`"logcumsumexp_backward" not implemented for 'Half'` for the native baseline.

The input_fn yields (inp, 1), so the scan runs along dim 1.

Only the autograd backward is timed; config() describes the forward call
because that is what the benchmark records for --base-data matching.
"""

import torch

OP_NAME = "logcumsumexp_backward"
DTYPES = [torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.logcumsumexp is unavailable on this torch build"

# yaml:GenericBenchmark2DOnly
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]

_DIM = 1


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


def _backward(out, xs, dout):
    """The timed region for an is_backward benchmark.

    base.Benchmark._benchmark_callable runs the forward once outside the timed
    region, draws dout = randn_like(out), collects the args that require grad
    and times only torch.autograd.grad(...). build_inputs() does the same
    preparation, so no forward work lands inside the measured call.
    """
    return torch.autograd.grad((out,), xs, grad_outputs=(dout,), retain_graph=True)


def native():
    if _resolve("torch.logcumsumexp") is None:
        return None
    return _backward


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)
    out = torch.logcumsumexp(inp, _DIM)
    dout = torch.randn_like(out)
    return (out, [inp], dout), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "requires_grad": True},
                {"scalar": _DIM},
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "dim": _DIM},
    }
