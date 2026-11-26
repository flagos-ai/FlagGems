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

"""NVIDIA native baseline for `quantized_max_pool2d`.

Mirrors benchmark/test_quantized_max_pool2d.py (QuantizedMaxPool2dBenchmark, a
GenericBenchmark subclass):
    op_name  = "quantized_max_pool2d"
    torch_op = torch.quantized_max_pool2d
    dtypes   = [torch.qint8]

The class overrides get_input_iter to walk POOL_SHAPES directly, so
core_shapes.yaml is not consulted. At --level core `_pool_configs` yields a
single (kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) pool;
the other configs are COMPREHENSIVE-only. Dilation stays 1 because the
accelerator baseline is cuDNN-backed and rejects anything else.

The input is a per-tensor quantized tensor built on the device (scale 0.1,
zero_point 0 for qint8); it must live on the accelerator or the native CPU
kernel would handle it instead. All pooling parameters are keywords.
"""

import torch

OP_NAME = "quantized_max_pool2d"
DTYPES = [torch.qint8]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.quantized_max_pool2d is unavailable on this torch build"
)

# test_quantized_max_pool2d.py::POOL_SHAPES -- ResNet-like NCHW feature maps
_SHAPES = [
    (4, 3, 224, 224),
    (16, 64, 56, 56),
    (32, 128, 28, 28),
    (64, 256, 14, 14),
    (128, 512, 7, 7),
]

_SCALE = 0.1
_KERNEL_SIZE = 3
_STRIDE = 2
_PADDING = 1
_DILATION = 1
_CEIL_MODE = False


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


def _quantized_input(shape, dtype, device):
    """benchmark::_quantized_input -- per-tensor quantized input on device."""
    if dtype == torch.quint8:
        data = torch.rand(shape, device=device) * 255.0
        zero_point = 128
    else:
        data = torch.randint(-128, 128, shape, device=device).float()
        zero_point = 0
    return torch.quantize_per_tensor(data, _SCALE, zero_point, dtype), zero_point


def native():
    op = _resolve("torch.quantized_max_pool2d")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp, _ = _quantized_input(tuple(binding["shape"]), dtype, device)
    return (inp,), {
        "kernel_size": _KERNEL_SIZE,
        "stride": _STRIDE,
        "padding": _PADDING,
        "dilation": _DILATION,
        "ceil_mode": _CEIL_MODE,
    }


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    n, c, h, w = shape
    h_out = (h + 2 * _PADDING - _KERNEL_SIZE) // _STRIDE + 1
    w_out = (w + 2 * _PADDING - _KERNEL_SIZE) // _STRIDE + 1
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt, "note": "per-tensor quantized"}],
            "kwargs": {
                "kernel_size": {"scalar": _KERNEL_SIZE},
                "stride": {"scalar": _STRIDE},
                "padding": {"scalar": _PADDING},
                "dilation": {"scalar": _DILATION},
                "ceil_mode": {"scalar": _CEIL_MODE},
            },
        },
        "outputs": {"out": {"shape": [n, c, h_out, w_out], "dtype": dt}},
        "dims": {
            "N": n,
            "C": c,
            "H": h,
            "W": w,
            "kernel_size": _KERNEL_SIZE,
            "stride": _STRIDE,
            "padding": _PADDING,
            "scale": _SCALE,
        },
    }
