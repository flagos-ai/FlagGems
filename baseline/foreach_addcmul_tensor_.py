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


"""NVIDIA native baseline for `foreach_addcmul_tensor_`.

Mirrors benchmark/test_foreach_ops.py (ForeachOpsBenchmark,
parametrized over the flag_gems _foreach_binary / _foreach_reduction
registration tables):
    op_name  = "foreach_addcmul_tensor_"
    torch_op = torch.ops.aten._foreach_addcmul_.Tensor
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:ForeachOpsBenchmark (benchmark/core_shapes.yaml)

ATen key `_foreach_addcmul_.Tensor`; each shape becomes a
TensorList of 16 tensors.

The Tensor overload takes its value tensor on the CPU; the
benchmark builds it with torch.tensor([0.5] * 16) and no device.
"""

import torch

OP_NAME = "foreach_addcmul_tensor_"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._foreach_addcmul_.Tensor is unavailable on this torch build"
)

# Shapes mirror the matching benchmark entry; see module docstring.
_SHAPES = [
    (256, 256),
    (512, 512),
    (1024, 1024),
]


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator the build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


_LIST_LENGTH = 16


def _tensor_list(shape, dtype, device):
    """benchmark/test_foreach_ops.py::ForeachOpsBenchmark.get_input_iter.

    Note the +1.0 offset: the shared iterator uses ``torch.rand(...) + 1.0`` so
    that div / pow / rsqrt-style operators stay in a well-conditioned range.
    """
    return [
        torch.rand(shape, device=device, dtype=dtype) + 1.0 for _ in range(_LIST_LENGTH)
    ]


def native():
    op = _resolve("torch.ops.aten._foreach_addcmul_.Tensor")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    tensors = _tensor_list(shape, dtype, device)
    tensor1 = _tensor_list(shape, dtype, device)
    tensor2 = _tensor_list(shape, dtype, device)
    # ATen wants the value tensor on the CPU for this overload.
    value = torch.tensor([0.5] * _LIST_LENGTH)
    return (tensors, tensor1, tensor2, value), {}


def key_shape(binding):
    # shape_desc is "M, N (per tensor, list of 16)"; the per-tensor shape plus
    # the fixed list length identifies the collection point.
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    tensors = [{"shape": shape, "dtype": dt} for _ in range(_LIST_LENGTH)]
    return {
        "inputs": {
            "args": [
                tensors,
                tensors,
                tensors,
                {"shape": [_LIST_LENGTH], "dtype": "torch.float32"},
            ],
            "kwargs": {},
        },
        "outputs": {"out": "same as input (written in place)"},
        "dims": {"shape": shape, "list_length": _LIST_LENGTH},
    }
