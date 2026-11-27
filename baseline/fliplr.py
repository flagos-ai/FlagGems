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

"""NVIDIA native baseline for `fliplr`.

Mirrors benchmark/test_fliplr.py (FliplrBenchmark, a base.Benchmark subclass):
    op_name  = "fliplr"
    torch_op = torch.fliplr
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with FLIPLR_SHAPES, so core_shapes.yaml is not
consulted. get_input_iter yields (inp, {}) -- the trailing empty dict is an
empty kwargs marker, so the call is op(inp).
"""

import torch

OP_NAME = "fliplr"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.fliplr is unavailable on this torch build"

# test_fliplr.py::FLIPLR_SHAPES
_SHAPES = [
    (4, 4),
    (64, 64),
    (512, 512),
    (1024, 4096),
    (4096, 4096),
    (64, 512, 512),
    (16, 128, 64, 64),
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
    op = _resolve("torch.fliplr")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = _generate_tensor_input(tuple(binding["shape"]), dtype, device)
    return (inp,), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {"args": [{"shape": shape, "dtype": dt}], "kwargs": {}},
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }
