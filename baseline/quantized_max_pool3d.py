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

"""NVIDIA native baseline for `quantized_max_pool3d`.

Mirrors benchmark/test_quantized_max_pool3d.py (QuantizedMaxPool3dBenchmark, a
GenericBenchmark subclass):
    op_name  = "quantized_max_pool3d"
    torch_op = _torch_op (the module-local reference below)
    dtypes   = [torch.quint8]

The class overrides get_input_iter to walk QUANTIZED_MAX_POOL3D_SHAPES, so
core_shapes.yaml is not consulted. At --level core one pooling config is
yielded (kernel_size 3, stride 2, padding 1, dilation 1, ceil_mode False);
the non-cubic and ceil_mode variants are COMPREHENSIVE-only.

CAUTION: PyTorch ships no QuantizedCUDA kernel for this operator, so the
reference necessarily runs on QuantizedCPU. The benchmark takes the host copy
of the input once in its input_fn -- outside the timed region -- so the
reported baseline is the CPU kernel alone rather than a device-to-host
transfer. That split is reproduced here: build_inputs attaches the CPU twin and
the timed callable only pools. See Q&A.md.
"""

import torch

OP_NAME = "quantized_max_pool3d"
DTYPES = [torch.quint8]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.quantized_max_pool3d is unavailable on this torch build"
)

# test_quantized_max_pool3d.py::QUANTIZED_MAX_POOL3D_SHAPES
_SHAPES = [
    (4, 3, 16, 56, 56),
    (8, 64, 8, 28, 28),
    (16, 128, 4, 14, 14),
    (32, 256, 2, 7, 7),
]

_QDTYPE = torch.quint8
_SCALE = 0.1
_ZERO_POINT = 128
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


def _cpu_input(qx):
    """benchmark::_cpu_input -- the CPU copy the input function prepared."""
    twin = getattr(qx, "_cpu_twin", None)
    if twin is None:
        twin = qx.to("cpu") if qx.device.type != "cpu" else qx
    return twin


def _torch_op(qx, **kwargs):
    """benchmark::_torch_op -- QuantizedCPU reference, host copy taken already."""
    return torch.quantized_max_pool3d(_cpu_input(qx), **kwargs)


def native():
    if _resolve("torch.quantized_max_pool3d") is None:
        return None
    return _torch_op


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    x = torch.randn(shape, device=device)
    qx = torch.quantize_per_tensor(x, scale=_SCALE, zero_point=_ZERO_POINT, dtype=dtype)
    # The host copy is taken here, outside the timed region.
    qx._cpu_twin = qx.to("cpu") if qx.device.type != "cpu" else qx
    return (qx,), {
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
    n, c, d, h, w = shape
    dt = str(dtype)

    def out_dim(size):
        return (size + 2 * _PADDING - _KERNEL_SIZE) // _STRIDE + 1

    out_shape = [n, c, out_dim(d), out_dim(h), out_dim(w)]
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
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {
            "N": n,
            "C": c,
            "D": d,
            "H": h,
            "W": w,
            "kernel_size": _KERNEL_SIZE,
            "stride": _STRIDE,
            "padding": _PADDING,
            "scale": _SCALE,
            "zero_point": _ZERO_POINT,
        },
    }
