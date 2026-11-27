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

"""NVIDIA native baseline for `nll_loss_nd_forward`.

Mirrors benchmark/test_nll_loss_nd.py (NLLLossNDBenchmark, a GenericBenchmark
subclass):
    op_name  = "nll_loss_nd_forward"
    torch_op = torch.nn.functional.nll_loss
    dtypes   = consts.FLOAT_DTYPES

The class overrides get_input_iter with its own eight-entry shape list and
ignores self.shapes, so core_shapes.yaml is not consulted. The list spans 2-D
through 5-D inputs.

The input is passed through log_softmax(dim=1) first, as the operator expects
log-probabilities. The target drops dim 1 and is drawn from randint(0, C) in
int64, where C = shape[1]. At --level core no weight / ignore_index /
reduction keywords are passed.
"""

import torch

OP_NAME = "nll_loss_nd_forward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.nn.functional.nll_loss is unavailable on this torch build"
)

# NLLLossNDBenchmark.get_input_iter::shapes
_SHAPES = [
    (64, 64),
    (256, 256),
    (10000, 65536),
    (32, 128, 512),
    (64, 64, 4, 8),
    (256, 256, 4, 8),
    (4096, 4096, 4, 8),
    (64, 64, 8, 4, 8),
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
    op = _resolve("torch.nn.functional.nll_loss")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _generate_tensor_input(shape, dtype, device)
    inp = torch.nn.functional.log_softmax(inp, dim=1)
    target_shape = list(shape)
    del target_shape[1]
    channels = shape[1]
    target = torch.randint(0, channels, target_shape, dtype=torch.long, device=device)
    return (inp, target), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    target_shape = list(shape)
    del target_shape[1]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "log-probabilities"},
                {"shape": target_shape, "dtype": "torch.int64"},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [], "dtype": dt}},
        "dims": {"shape": shape, "C": shape[1], "target_shape": target_shape},
    }
