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

"""NVIDIA native baseline for `linalg_qr`.

Mirrors benchmark/test_linalg_qr.py (QRGenericBenchmark, a GenericBenchmark
subclass):
    op_name  = "linalg_qr"
    torch_op = torch.ops.aten.linalg_qr
    dtypes   = [float32] + [float64] when the backend has FP64
    shapes   = yaml:linalg_qr (benchmark/core_shapes.yaml)

The class returns [] from set_more_shapes because the generic 1-D extras are
not valid QR inputs, so only the yaml shapes apply.

The test is parametrized over QR_MODES = ["reduced", "complete", "r"] under
one op_name, so each shape contributes three collection points; the grid below
carries the mode explicitly. mode is passed as a keyword.
"""

import torch

OP_NAME = "linalg_qr"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.linalg_qr is unavailable on this torch build"
)

# yaml:linalg_qr
_SHAPES = [
    (8, 8),
    (64, 64),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (128, 32),
    (512, 64),
    (4096, 4),
    (8192, 8),
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256),
    (64, 8, 8),
    (128, 32, 32),
    (32, 128, 128),
    (4, 1024, 1024),
]

# test_linalg_qr.py::QR_MODES
_MODES = ("reduced", "complete", "r")


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
    op = _resolve("torch.ops.aten.linalg_qr")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s), "mode": mode} for s in _SHAPES for mode in _MODES]


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (inp,), {"mode": binding["mode"]}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_{binding['mode']}"


def config(binding, dtype):
    shape = list(binding["shape"])
    mode = binding["mode"]
    batch, m, n = shape[:-2], shape[-2], shape[-1]
    k = min(m, n)
    dt = str(dtype)
    # benchmark::_qr_out_shapes
    if mode == "complete":
        q_shape, r_shape = batch + [m, m], batch + [m, n]
    elif mode == "r":
        q_shape, r_shape = [0], batch + [k, n]
    else:
        q_shape, r_shape = batch + [m, k], batch + [k, n]
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {"mode": mode},
        },
        "outputs": {
            "Q": {"shape": q_shape, "dtype": dt},
            "R": {"shape": r_shape, "dtype": dt},
        },
        "dims": {"shape": shape, "M": m, "N": n, "mode": mode},
    }
