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

"""NVIDIA native baseline for `choose_qparams_per_tensor`.

Mirrors benchmark/test_choose_qparams_per_tensor.py
(ChooseQParamsPerTensorBenchmark, a GenericBenchmark subclass using the
two-phase case_fn/build_inputs_fn pair):
    op_name  = "choose_qparams_per_tensor"
    torch_op = torch.ops.aten._choose_qparams_per_tensor.default
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own CQPT_SHAPES list, deliberately
bypassing core_shapes.yaml: the base fallback includes a 1 B-element tensor
whose allocation cost would dominate this whole-tensor min/max reduction.

_case_fn expands each shape over reduce_range in (False, True) and
_build_inputs_fn passes it as a keyword, so every shape yields two collection
points: op(input, reduce_range=...).
"""

import torch

OP_NAME = "choose_qparams_per_tensor"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._choose_qparams_per_tensor.default is unavailable on this build"
)

# test_choose_qparams_per_tensor.py::CQPT_SHAPES
_SHAPES = [
    (65536,),
    (1_048_576,),
    (4096, 1024),
]

_REDUCE_RANGE = (False, True)

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
    op = _resolve("torch.ops.aten._choose_qparams_per_tensor.default")
    return op if callable(op) else None


def grid():
    return [
        {"shape": list(s), "reduce_range": reduce_range}
        for s in _SHAPES
        for reduce_range in _REDUCE_RANGE
    ]


def build_inputs(binding, dtype, device):
    inp = _generate_tensor_input(tuple(binding["shape"]), dtype, device)
    return (inp,), {"reduce_range": binding["reduce_range"]}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_reduce{int(binding['reduce_range'])}"


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {"reduce_range": {"scalar": binding["reduce_range"]}},
        },
        "outputs": {
            "scale": {"shape": [], "dtype": "torch.float64"},
            "zero_point": {"shape": [], "dtype": "torch.int64"},
        },
        "dims": {"shape": shape, "reduce_range": binding["reduce_range"]},
    }
