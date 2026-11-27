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


"""NVIDIA native baseline for `foreach_zero`.

Mirrors benchmark/test_foreach_ops.py (ForeachOpsBenchmark,
parametrized over the flag_gems _foreach_binary / _foreach_reduction
registration tables):
    op_name  = "foreach_zero"
    torch_op = torch.ops.aten._foreach_zero
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:ForeachOpsBenchmark (benchmark/core_shapes.yaml)

ATen key `_foreach_zero`; each shape becomes a
TensorList of 16 tensors.
"""

import torch

OP_NAME = "foreach_zero"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._foreach_zero is unavailable on this torch build"
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
    op = _resolve("torch.ops.aten._foreach_zero")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    tensors = _tensor_list(shape, dtype, device)
    return (tensors,), {}


def key_shape(binding):
    # shape_desc is "M, N (per tensor, list of 16)"; the per-tensor shape plus
    # the fixed list length identifies the collection point.
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    tensors = [{"shape": shape, "dtype": dt} for _ in range(_LIST_LENGTH)]
    return {
        "inputs": {"args": [tensors], "kwargs": {}},
        "outputs": {"out": tensors},
        "dims": {"shape": shape, "list_length": _LIST_LENGTH},
    }
