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


"""NVIDIA native baseline for `logit_backward`.

Mirrors benchmark/test_logit_backward.py (base.UnaryPointwiseBenchmark):
    op_name  = "logit_backward"
    torch_op = lambda a: torch.ops.aten.logit_backward(torch.ones_like(a), torch.rand_like(a), 1e-06)
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:UnaryPointwiseBenchmark (benchmark/core_shapes.yaml)

Despite the name this is a forward-only benchmark (is_backward is not set):
the aten logit_backward kernel is called directly on freshly allocated
grad/self tensors.
"""

import torch

OP_NAME = "logit_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.logit_backward is unavailable on this torch build"
)

# Shapes mirror the matching benchmark entry; see module docstring.
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


_EPS = 1e-6


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name> raises instead of returning None for an operator the
    build does not carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _logit_backward(a):
    """benchmark torch_op:

        lambda a: torch.ops.aten.logit_backward(
            torch.ones_like(a), torch.rand_like(a), 1e-6
        )

    The two auxiliary tensors are allocated inside the timed callable by the
    benchmark, so they are allocated here as well rather than in build_inputs.
    """
    return torch.ops.aten.logit_backward(torch.ones_like(a), torch.rand_like(a), _EPS)


def native():
    if _resolve("torch.ops.aten.logit_backward") is None:
        return None
    return _logit_backward


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _generate_tensor_input(shape, dtype, device)
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
