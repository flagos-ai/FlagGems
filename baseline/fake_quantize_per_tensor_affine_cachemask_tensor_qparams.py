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

"""NVIDIA native baseline for
`fake_quantize_per_tensor_affine_cachemask_tensor_qparams`.

Mirrors
benchmark/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py
(base.GenericBenchmark):
    op_name  = "fake_quantize_per_tensor_affine_cachemask_tensor_qparams"
    torch_op = torch.ops.aten
                   ._fake_quantize_per_tensor_affine_cachemask_tensor_qparams
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

This is the tensor-qparams overload: scale, zero_point and fake_quant_enabled
are 0-d tensors (float32 / int32 / int64) rather than Python scalars.
quant_min/quant_max are -128/127, the int8 range.
"""

import torch

OP_NAME = "fake_quantize_per_tensor_affine_cachemask_tensor_qparams"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams "
    "is unavailable on this torch build"
)

# yaml:Benchmark -- the shared default shape list
_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

_QUANT_MIN = -128
_QUANT_MAX = 127


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
    op = _resolve(
        "torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams"
    )
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device)
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    zero_point = torch.tensor(0, dtype=torch.int32, device=device)
    fake_quant_enabled = torch.tensor(1, dtype=torch.int64, device=device)
    return (
        inp,
        scale,
        zero_point,
        fake_quant_enabled,
        _QUANT_MIN,
        _QUANT_MAX,
    ), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": [], "dtype": "torch.float32"},
                {"shape": [], "dtype": "torch.int32"},
                {"shape": [], "dtype": "torch.int64"},
                {"scalar": _QUANT_MIN},
                {"scalar": _QUANT_MAX},
            ],
            "kwargs": {},
        },
        "outputs": {
            "out": {"shape": shape, "dtype": dt},
            "mask": {"shape": shape, "dtype": "torch.bool"},
        },
        "dims": {
            "shape": shape,
            "quant_min": _QUANT_MIN,
            "quant_max": _QUANT_MAX,
        },
    }
