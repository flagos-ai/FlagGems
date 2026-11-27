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

"""NVIDIA native baseline for `slow_conv_dilated3d_out`.

Mirrors benchmark/test_slow_conv_dilated3d.py (SlowConvDilated3dOutBenchmark, a
SlowConvDilated3dBenchmark subclass with with_out=True):
    op_name  = "slow_conv_dilated3d_out"
    torch_op = torch.ops.aten.slow_conv_dilated3d.out
    dtypes   = consts.FLOAT_DTYPES (one parametrized run each)

The parent class overrides set_shapes with SCC3D_CASES and declares shape_desc
"N, C_in, D, H, W, C_out, kD, kH, kW, stride, padding", so core_shapes.yaml is
not consulted.

dilation is [1, 1, 1] for every case. The out buffer's spatial dims follow the
benchmark's own formula, which assumes dilation 1:
    out = (in + 2 * padding - (k - 1) - 1) // stride + 1
"""

import torch

OP_NAME = "slow_conv_dilated3d_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.slow_conv_dilated3d.out is unavailable on this torch build"
)

# test_slow_conv_dilated3d.py::SCC3D_CASES
_SHAPES = [
    (2, 16, 16, 16, 16, 32, 3, 3, 3, 1, 1),
    (1, 32, 12, 12, 12, 64, 3, 3, 3, 1, 1),
    (2, 8, 24, 24, 24, 16, 3, 3, 3, 2, 1),
    (1, 64, 16, 16, 16, 64, 1, 1, 1, 1, 0),
    (2, 4, 24, 24, 24, 8, 2, 2, 2, 1, 0),
]

_FIELDS = (
    "n",
    "c_in",
    "d",
    "h",
    "w",
    "c_out",
    "kd",
    "kh",
    "kw",
    "stride",
    "padding",
)

_DILATION = [1, 1, 1]


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


def _out_spatial(size, kernel, stride, padding):
    return (size + 2 * padding - (kernel - 1) - 1) // stride + 1


def native():
    op = _resolve("torch.ops.aten.slow_conv_dilated3d.out")
    return op if callable(op) else None


def grid():
    return [dict(zip(_FIELDS, case)) for case in _SHAPES]


def build_inputs(binding, dtype, device):
    n, c_in = binding["n"], binding["c_in"]
    d, h, w = binding["d"], binding["h"], binding["w"]
    c_out = binding["c_out"]
    kd, kh, kw = binding["kd"], binding["kh"], binding["kw"]
    stride, padding = binding["stride"], binding["padding"]
    inp = torch.randn((n, c_in, d, h, w), dtype=dtype, device=device)
    weight = torch.randn((c_out, c_in, kd, kh, kw), dtype=dtype, device=device)
    bias = torch.randn((c_out,), dtype=dtype, device=device)
    out = torch.empty(
        (
            n,
            c_out,
            _out_spatial(d, kd, stride, padding),
            _out_spatial(h, kh, stride, padding),
            _out_spatial(w, kw, stride, padding),
        ),
        dtype=dtype,
        device=device,
    )
    return (
        inp,
        weight,
        [kd, kh, kw],
        bias,
        [stride, stride, stride],
        [padding, padding, padding],
        list(_DILATION),
    ), {"out": out}


def key_shape(binding):
    return [binding[key] for key in _FIELDS]


def config(binding, dtype):
    n, c_in = binding["n"], binding["c_in"]
    d, h, w = binding["d"], binding["h"], binding["w"]
    c_out = binding["c_out"]
    kd, kh, kw = binding["kd"], binding["kh"], binding["kw"]
    stride, padding = binding["stride"], binding["padding"]
    out_shape = [
        n,
        c_out,
        _out_spatial(d, kd, stride, padding),
        _out_spatial(h, kh, stride, padding),
        _out_spatial(w, kw, stride, padding),
    ]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [n, c_in, d, h, w], "dtype": dt},
                {"shape": [c_out, c_in, kd, kh, kw], "dtype": dt},
                [kd, kh, kw],
                {"shape": [c_out], "dtype": dt},
                [stride, stride, stride],
                [padding, padding, padding],
                list(_DILATION),
            ],
            "kwargs": {"out": {"shape": out_shape, "dtype": dt}},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {key: binding[key] for key in _FIELDS},
    }
