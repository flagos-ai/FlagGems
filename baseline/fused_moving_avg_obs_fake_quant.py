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

"""NVIDIA native baseline for `fused_moving_avg_obs_fake_quant`.

Mirrors benchmark/test_fused_moving_avg_obs_fake_quant.py
(FusedMovingAvgObsFakeQuantBenchmark, a GenericBenchmark subclass):
    op_name  = "fused_moving_avg_obs_fake_quant"
    torch_op = torch_fused_moving_avg_obs_fake_quant, a pass-through to
               torch.fused_moving_avg_obs_fake_quant
    dtypes   = [torch.float32]

The class's set_more_shapes *assigns* self.shapes and then returns [], so the
five large per-tensor QAT shapes below replace whatever core_shapes.yaml
provided -- but only at --level comprehensive, because set_more_shapes is
called only then. At --level core the shapes are the generic yaml:Benchmark
list.

This baseline uses the class's own five shapes: they are the ones the class
documents as realistic for per-tensor QAT (activation tensors of >=128M
elements), and the author's intent is clear from the assignment. See Q&A.md.

float32 only: the ATen operator supports no other dtype. observer_on and
fake_quant_on are both enabled, averaging_const 0.01, quant range 0..255,
ch_axis -1, per_row_fake_quant and symmetric_quant both False.
running_min/running_max/scale/zero_point are updated in place.
"""

import torch

OP_NAME = "fused_moving_avg_obs_fake_quant"
DTYPES = [torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (3, 4, 5, 6)
NATIVE_UNAVAILABLE_REASON = (
    "torch.fused_moving_avg_obs_fake_quant is unavailable on this torch build"
)

# FusedMovingAvgObsFakeQuantBenchmark.set_more_shapes
_SHAPES = [
    (1 << 27,),  # 128M -- e.g. batch=16, seq=1024, hidden=8192
    (1 << 28,),  # 256M
    (1 << 29,),  # 512M
    (1024, 1024, 1024),  # 1G 3D
    (1 << 30,),  # 1G 1D
]

_AVERAGING_CONST = 0.01
_QUANT_MIN = 0
_QUANT_MAX = 255
_CH_AXIS = -1
_PER_ROW_FAKE_QUANT = False
_SYMMETRIC_QUANT = False


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
    op = _resolve("torch.fused_moving_avg_obs_fake_quant")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    observer_on = torch.tensor([1], dtype=torch.int32, device=device)
    fake_quant_on = torch.tensor([1], dtype=torch.int32, device=device)
    running_min = torch.tensor([-0.5], dtype=torch.float32, device=device)
    running_max = torch.tensor([0.5], dtype=torch.float32, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    zero_point = torch.zeros(1, dtype=torch.int32, device=device)
    return (
        inp,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        _AVERAGING_CONST,
        _QUANT_MIN,
        _QUANT_MAX,
        _CH_AXIS,
        _PER_ROW_FAKE_QUANT,
        _SYMMETRIC_QUANT,
    ), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    f32 = "torch.float32"
    i32 = "torch.int32"
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": f32},
                {"shape": [1], "dtype": i32},
                {"shape": [1], "dtype": i32},
                {"shape": [1], "dtype": f32, "note": "updated in place"},
                {"shape": [1], "dtype": f32, "note": "updated in place"},
                {"shape": [1], "dtype": f32, "note": "updated in place"},
                {"shape": [1], "dtype": i32, "note": "updated in place"},
                {"scalar": _AVERAGING_CONST},
                {"scalar": _QUANT_MIN},
                {"scalar": _QUANT_MAX},
                {"scalar": _CH_AXIS},
                {"scalar": _PER_ROW_FAKE_QUANT},
                {"scalar": _SYMMETRIC_QUANT},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": f32}},
        "dims": {
            "shape": shape,
            "averaging_const": _AVERAGING_CONST,
            "quant_min": _QUANT_MIN,
            "quant_max": _QUANT_MAX,
            "ch_axis": _CH_AXIS,
        },
    }
