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

"""NVIDIA native baseline for `special_scaled_modified_bessel_k0_out`.

Mirrors benchmark/test_special_scaled_modified_bessel_k0.py
(SpecialScaledModifiedBesselK0OutBenchmark, a GenericBenchmark subclass using
the two-phase case_fn/build_inputs_fn pair):
    op_name  = "special_scaled_modified_bessel_k0_out"
    torch_op = torch.ops.aten.special_scaled_modified_bessel_k0.out
    dtypes   = [torch.float32]

The parent class overrides set_shapes with its curated `_SHAPES` list,
deliberately bypassing core_shapes.yaml: there is no entry for this operator
and the base fallback holds a 1e9-element tensor.

The out buffer takes the promoted result dtype -- the input dtype for a
floating input, float32 otherwise -- and is allocated in build_inputs so it
stays outside the timed region. With the float32-only dtype list the two
coincide.

float64 is in the dtype list because the benchmark appends it when
flag_gems.runtime.device.support_fp64 is true, and the NVIDIA backend
reports fp64_enabled=True.
"""

import torch

OP_NAME = "special_scaled_modified_bessel_k0_out"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.special_scaled_modified_bessel_k0.out is unavailable"
)

# test_special_scaled_modified_bessel_k0.py::_SHAPES
_SHAPES = [
    (1024,),
    (65536,),
    (64, 512, 512),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
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
    op = _resolve("torch.ops.aten.special_scaled_modified_bessel_k0.out")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    if dtype == torch.float64:
        # generate_tensor_input has no float64 branch.
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        inp = _generate_tensor_input(shape, dtype, device)
    out_dtype = dtype if dtype.is_floating_point else torch.float32
    out = torch.empty(shape, dtype=out_dtype, device=device)
    return (inp,), {"out": out}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    out_dt = dt if dtype.is_floating_point else "torch.float32"
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {"out": {"shape": shape, "dtype": out_dt}},
        },
        "outputs": {"out": {"shape": shape, "dtype": out_dt}},
        "dims": {"shape": shape},
    }
