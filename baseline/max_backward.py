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


"""NVIDIA native baseline for `max_backward`.

Mirrors benchmark/test_max.py (base.UnaryReductionBenchmark,
is_backward=True):
    op_name  = "max_backward"
    torch_op = _max_backward_values (the module-local reference below)
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:UnaryReductionBenchmark, then filtered by
               Benchmark.MAX_BACKWARD_ELEMENTS (2**27).

The reduction yields (inp, 1) for every multi-dimensional shape and
(inp,) for the 1-D shape, so one grid covers both the torch.max.dim
adjoint and the global torch.max tie-splitting adjoint. Only the
autograd backward is timed.
"""

import torch

OP_NAME = "max_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.max is unavailable on this torch build"

# Shapes mirror the matching benchmark entry; see module docstring.
_SHAPES = [
    (1048576,),
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


def _reduce_values(inp, dim=None):
    """benchmark/test_max.py::_max_backward_values -- the native reference.

    def _max_backward_values(inp, dim=None):
        if dim is None:
            return torch.max(inp)
        return torch.max(inp, dim=dim).values
    """
    if dim is None:
        return torch.max(inp)
    return torch.max(inp, dim=dim).values


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
    if _resolve("torch.max") is None:
        return None
    return _backward


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _generate_tensor_input(shape, dtype, device).requires_grad_(True)
    # UnaryReductionBenchmark yields (inp, 1) for ndim > 1 and (inp,) otherwise.
    out = _reduce_values(inp, 1) if inp.ndim > 1 else _reduce_values(inp)
    dout = torch.randn_like(out)
    return (out, [inp], dout), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    args = [{"shape": shape, "dtype": dt, "requires_grad": True}]
    if len(shape) > 1:
        args.append(1)
    return {
        "inputs": {"args": args, "kwargs": {}},
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "dim": 1 if len(shape) > 1 else None},
    }
