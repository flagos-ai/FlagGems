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

"""NVIDIA native baseline for `upsample_bilinear2d_aa`.

Mirrors benchmark/test_upsample_bilinear2d_aa.py (UpsampleBenchmark, a
base.Benchmark subclass):
    op_name  = "upsample_bilinear2d_aa"
    torch_op = torch._C._nn._upsample_bilinear2d_aa
    dtypes   = consts.FLOAT_DTYPES on NVIDIA (the test narrows the list only
               for mthreads and kunlunxin)
    shapes   = yaml:UpsampleBenchmark (benchmark/core_shapes.yaml),
               shape_desc "N, C, H, W"

The output size is twice the input in both spatial dimensions (scale_factors
(2, 2)), align_corners is False and scales_h/scales_w are None. Every argument
is a keyword, as the input_fn yields them.
"""

import torch

OP_NAME = "upsample_bilinear2d_aa"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch._C._nn._upsample_bilinear2d_aa is unavailable on this torch build"
)

# yaml:UpsampleBenchmark -- shape_desc "N, C, H, W"
_SHAPES = [
    (1, 3, 512, 512),
    (8, 16, 128, 128),
    (2, 3, 1024, 1024),
    (16, 16, 512, 512),
    (16, 16, 1024, 1024),
]

_SCALE_FACTORS = (2, 2)
_ALIGN_CORNERS = False


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
    op = _resolve("torch._C._nn._upsample_bilinear2d_aa")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    _, _, height, width = shape
    inp = torch.randn(size=shape, device=device, dtype=dtype)
    output_size = (
        int(height * _SCALE_FACTORS[0]),
        int(width * _SCALE_FACTORS[1]),
    )
    return (), {
        "input": inp,
        "output_size": output_size,
        "align_corners": _ALIGN_CORNERS,
        "scales_h": None,
        "scales_w": None,
    }


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    n, c, height, width = shape
    output_size = [
        int(height * _SCALE_FACTORS[0]),
        int(width * _SCALE_FACTORS[1]),
    ]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [],
            "kwargs": {
                "input": {"shape": shape, "dtype": dt},
                "output_size": output_size,
                "align_corners": {"scalar": _ALIGN_CORNERS},
                "scales_h": None,
                "scales_w": None,
            },
        },
        "outputs": {"out": {"shape": [n, c] + output_size, "dtype": dt}},
        "dims": {
            "N": n,
            "C": c,
            "H": height,
            "W": width,
            "output_size": output_size,
        },
    }
