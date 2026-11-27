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

"""NVIDIA native baseline for `combinations`.

Mirrors benchmark/test_combinations.py (CombinationsBenchmark, a base.Benchmark
subclass):
    op_name  = "combinations"
    torch_op = torch.combinations
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with COMBINATIONS_CASES and shape_desc
"input length, r, with_replacement", so core_shapes.yaml is not consulted.
Each case builds a 1-D tensor of `n` elements and calls
torch.combinations(inp, r, with_replacement).
"""

import torch

OP_NAME = "combinations"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.combinations is unavailable on this torch build"

# test_combinations.py::COMBINATIONS_CASES -- (input length, r, with_replacement)
_SHAPES = [
    (1024, 2, False),
    (1024, 2, True),
    (256, 3, False),
    (256, 3, True),
    (128, 4, False),
    (32, 5, False),
    (16, 6, True),
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
    op = _resolve("torch.combinations")
    return op if callable(op) else None


def grid():
    return [
        {"n": n, "r": r, "with_replacement": with_replacement}
        for n, r, with_replacement in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    inp = _generate_tensor_input((binding["n"],), dtype, device)
    return (inp, binding["r"], binding["with_replacement"]), {}


def key_shape(binding):
    return f"n{binding['n']}_r{binding['r']}" f"_repl{int(binding['with_replacement'])}"


def config(binding, dtype):
    n, r = binding["n"], binding["r"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [n], "dtype": dt},
                {"scalar": r},
                {"scalar": binding["with_replacement"]},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"dtype": dt, "note": "C(n, r) or C(n+r-1, r) rows"}},
        "dims": {
            "n": n,
            "r": r,
            "with_replacement": binding["with_replacement"],
        },
    }
