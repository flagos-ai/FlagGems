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

"""NVIDIA native baseline for `fmod_tensor_`.

Mirrors benchmark/test_fmod.py (base.GenericBenchmark):
    op_name  = "fmod_tensor_"
    torch_op = torch.Tensor.fmod_
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

The divisor has its zeros replaced by ones
(`torch.where(inp2 == 0, ones_like(inp2), inp2)`) so the modulus is defined
everywhere.

Note the benchmark passes `inplace=True`, not `is_inplace=True`. Benchmark
.__init__ only forwards a kwarg when the attribute already exists, and
`inplace` is not an attribute of Benchmark, so that flag is silently dropped
and the benchmark treats this in-place operator as out-of-place. IS_INPLACE is
True here so the collector restores the mutated input outside every timed
interval. See Q&A.md.
"""

import torch

OP_NAME = "fmod_tensor_"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)
NATIVE_UNAVAILABLE_REASON = "torch.Tensor.fmod_ is unavailable on this torch build"

# yaml:Benchmark -- the shared default shape list
_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
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


def native():
    op = _resolve("torch.Tensor.fmod_")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp1 = _generate_tensor_input(shape, dtype, device)
    inp2 = _generate_tensor_input(shape, dtype, device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    return (inp1, inp2), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "written in place"},
                {"shape": shape, "dtype": dt, "note": "zeros replaced by ones"},
            ],
            "kwargs": {},
        },
        "outputs": {"self": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }
