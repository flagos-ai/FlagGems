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

"""NVIDIA native baseline for `scatter_add`.

Mirrors benchmark/test_scatter_add.py (TensorSelectBenchmark, a
GenericBenchmark2DOnly subclass):
    op_name  = "scatter_add"
    torch_op = torch.scatter_add
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:TensorSelectBenchmark (benchmark/core_shapes.yaml)

scatter_input_fn builds dim = -1, an index of twice the input's last dimension
drawn from randint(0, shape[-1]) in int64, and a src 16 elements larger than
the index in every dimension -- scatter_add only requires src to be at least
as large as index, so the padding is deliberate slack.
"""

import torch

OP_NAME = "scatter_add"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.scatter_add is unavailable on this torch build"

# yaml:TensorSelectBenchmark
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]

_DIM = -1
_SRC_SLACK = 16


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
    op = _resolve("torch.scatter_add")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device)
    size_dim = shape[_DIM]
    index_shape = list(shape)
    index_shape[_DIM] = 2 * shape[_DIM]
    index = torch.randint(0, size_dim, index_shape, dtype=torch.long, device=device)
    src_shape = [size + _SRC_SLACK for size in index.shape]
    src = torch.randn(src_shape, dtype=dtype, device=device)
    return (inp, _DIM, index, src), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    index_shape = list(shape)
    index_shape[_DIM] = 2 * shape[_DIM]
    src_shape = [size + _SRC_SLACK for size in index_shape]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": _DIM},
                {"shape": index_shape, "dtype": "torch.int64"},
                {"shape": src_shape, "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {
            "shape": shape,
            "dim": _DIM,
            "index_shape": index_shape,
            "src_shape": src_shape,
        },
    }
