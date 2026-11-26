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

"""NVIDIA native baseline for `slow_conv_transpose3d`.

Mirrors benchmark/test_slow_conv_transpose3d.py (SlowConvTranspose3dBenchmark,
a base.Benchmark subclass):
    op_name  = "slow_conv_transpose3d"
    torch_op = _slow_conv_transpose3d_ref, a pass-through to
               torch.ops.aten.slow_conv_transpose3d
    dtypes   = consts.FLOAT_DTYPES (one parametrized run each)

The class overrides set_shapes with SCT3D_CASES, so core_shapes.yaml is not
consulted. Note the transposed-conv weight layout is (C_in, C_out, kD, kH, kW),
not (C_out, C_in, ...).

output_padding is [0, 0, 0] and dilation [1, 1, 1].
"""

import torch

OP_NAME = "slow_conv_transpose3d"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.slow_conv_transpose3d is unavailable on this torch build"
)

# test_slow_conv_transpose3d.py::SCT3D_CASES
_SHAPES = [
    (2, 4, 8, 16, 16, 8, 3, 3, 3, 1, 0),
    (1, 8, 6, 8, 8, 4, 2, 2, 2, 2, 0),
    (2, 8, 8, 16, 16, 16, 3, 3, 3, 2, 1),
    (1, 16, 8, 8, 8, 8, 3, 3, 3, 1, 1),
    (2, 16, 10, 10, 10, 32, 2, 2, 2, 2, 1),
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

_OUTPUT_PADDING = [0, 0, 0]
_DILATION = [1, 1, 1]

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


def _slow_conv_transpose3d_ref(*args):
    """benchmark::_slow_conv_transpose3d_ref."""
    return torch.ops.aten.slow_conv_transpose3d(*args)


def native():
    if _resolve("torch.ops.aten.slow_conv_transpose3d") is None:
        return None
    return _slow_conv_transpose3d_ref


def grid():
    return [dict(zip(_FIELDS, case)) for case in _SHAPES]


def build_inputs(binding, dtype, device):
    n, c_in = binding["n"], binding["c_in"]
    d, h, w = binding["d"], binding["h"], binding["w"]
    c_out = binding["c_out"]
    kd, kh, kw = binding["kd"], binding["kh"], binding["kw"]
    stride, padding = binding["stride"], binding["padding"]
    inp = _generate_tensor_input((n, c_in, d, h, w), dtype, device)
    # Transposed-conv weight layout: (C_in, C_out, kD, kH, kW).
    weight = _generate_tensor_input((c_in, c_out, kd, kh, kw), dtype, device)
    bias = _generate_tensor_input((c_out,), dtype, device)
    return (
        inp,
        weight,
        [kd, kh, kw],
        bias,
        [stride, stride, stride],
        [padding, padding, padding],
        list(_OUTPUT_PADDING),
        list(_DILATION),
    ), {}


def key_shape(binding):
    return [binding[key] for key in _FIELDS]


def config(binding, dtype):
    n, c_in = binding["n"], binding["c_in"]
    d, h, w = binding["d"], binding["h"], binding["w"]
    c_out = binding["c_out"]
    kd, kh, kw = binding["kd"], binding["kh"], binding["kw"]
    stride, padding = binding["stride"], binding["padding"]
    dt = str(dtype)

    def out_spatial(size, kernel):
        return (size - 1) * stride - 2 * padding + kernel

    out_shape = [
        n,
        c_out,
        out_spatial(d, kd),
        out_spatial(h, kh),
        out_spatial(w, kw),
    ]
    return {
        "inputs": {
            "args": [
                {"shape": [n, c_in, d, h, w], "dtype": dt},
                {"shape": [c_in, c_out, kd, kh, kw], "dtype": dt},
                [kd, kh, kw],
                {"shape": [c_out], "dtype": dt},
                [stride, stride, stride],
                [padding, padding, padding],
                list(_OUTPUT_PADDING),
                list(_DILATION),
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {key: binding[key] for key in _FIELDS},
    }
