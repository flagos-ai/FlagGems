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

"""NVIDIA native baseline for `feature_alpha_dropout_`.

Mirrors benchmark/test_feature_alpha_dropout_.py (FeatureAlphaDropoutInplaceBenchmark, a
GenericBenchmark subclass):
    op_name  = "feature_alpha_dropout_"
    torch_op = torch.ops.aten.feature_alpha_dropout_
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own list, so core_shapes.yaml is not
consulted. The input_fn yields (input, 0.5, True) -- p and train are fixed.

The benchmark leaves is_inplace unset although this overload writes back into
its input. IS_INPLACE is True here so the collector restores the input outside
every timed interval; repeated in-place dropout on one buffer otherwise zeroes
more and more channels across warmups.
"""

import torch

OP_NAME = "feature_alpha_dropout_"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.feature_alpha_dropout_ is unavailable on this torch build"
)

# FeatureAlphaDropout*Benchmark.set_shapes
_SHAPES = [
    (2, 3),
    (4, 8, 16),
    (8, 16, 32, 32),
    (16, 32, 64, 64),
    (32, 64, 64, 64),
    (32, 64, 128, 128),
    (4, 8, 16, 16, 16),
]

_P = 0.5
_TRAIN = True

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
    op = _resolve("torch.ops.aten.feature_alpha_dropout_")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = _generate_tensor_input(tuple(binding["shape"]), dtype, device)
    return (inp, _P, _TRAIN), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "written in place"},
                {"scalar": _P},
                {"scalar": _TRAIN},
            ],
            "kwargs": {},
        },
        "outputs": {"self": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "p": _P, "train": _TRAIN},
    }
