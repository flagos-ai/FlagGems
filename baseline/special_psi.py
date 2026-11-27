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

"""NVIDIA native baseline for `special_psi`.

Mirrors benchmark/test_special_psi.py (SpecialPsiBenchmark, a
base.UnaryPointwiseBenchmark subclass):
    op_name  = "special_psi"
    torch_op = torch.ops.aten.special_psi.default
    dtypes   = [float16, float32] + bfloat16 and float64 when the backend
               supports them -- on NVIDIA both do, so all four run
    shapes   = yaml:UnaryPointwiseBenchmark (benchmark/core_shapes.yaml)

consts.FLOAT_DTYPES leaves bfloat16 ungated and has no float64, so this
benchmark gates both on the backend's static capabilities instead.

The subclass overrides build_inputs only to add a float64 branch, because
benchmark/utils.py::generate_tensor_input has none; every other dtype keeps
the shared builder. The same split is reproduced here.
"""

import torch

OP_NAME = "special_psi"
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.special_psi.default is unavailable on this torch build"
)

# yaml:UnaryPointwiseBenchmark -- shape_desc "(B), M, N"
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
    op = _resolve("torch.ops.aten.special_psi.default")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    if dtype == torch.float64:
        # generate_tensor_input has no float64 branch.
        return (torch.randn(shape, dtype=torch.float64, device=device),), {}
    return (_generate_tensor_input(shape, dtype, device),), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }
