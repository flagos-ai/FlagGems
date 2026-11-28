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


"""NVIDIA native baseline for `log2_backward`.

Mirrors benchmark/test_log2.py (base.UnaryPointwiseBenchmark,
is_backward=True):
    op_name  = "log2" + "_backward" (appended by base.Benchmark)
    torch_op = torch.log2
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:UnaryPointwiseBenchmark, then filtered by
               Benchmark.MAX_BACKWARD_ELEMENTS (2**27), which drops the
               two 2**30-element shapes.

Only the autograd backward is timed; config() still describes the
forward call because that is what the benchmark records for
--base-data matching.
"""

import torch

OP_NAME = "log2_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.log2 is unavailable on this torch build"

# Shapes mirror the matching benchmark entry; see module docstring.
_SHAPES = [
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
]

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_INT_DTYPES = (torch.int16, torch.int32)
_BOOL_DTYPES = (torch.bool,)
_COMPLEX_DTYPES = (torch.complex64,)


def _generate_tensor_input(shape, dtype, device):
    """benchmark/utils.py::generate_tensor_input, verbatim semantics."""
    if dtype in _FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in _INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)
    if dtype in _BOOL_DTYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(device)
    if dtype in _COMPLEX_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    raise TypeError(f"unsupported dtype: {dtype}")


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent."""
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _backward(out, xs, dout):
    """The timed region for an is_backward benchmark.

    base.Benchmark._benchmark_callable builds `out` once, draws
    `dout = torch.randn_like(out)`, collects `xs` (the args requiring grad) and
    then times *only*::

        torch.autograd.grad((out,), xs, grad_outputs=(dout,), retain_graph=True)

    build_inputs() does the same preparation so the forward never lands inside
    the measured call, and retain_graph keeps the graph alive across the
    collector's warmup and sampling repetitions.
    """
    return torch.autograd.grad((out,), xs, grad_outputs=(dout,), retain_graph=True)


def native():
    if _resolve("torch.log2") is None:
        return None
    return _backward


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _generate_tensor_input(shape, dtype, device).requires_grad_(True)
    out = torch.log2(inp)
    dout = torch.randn_like(out)
    return (out, [inp], dout), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt, "requires_grad": True}],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }
