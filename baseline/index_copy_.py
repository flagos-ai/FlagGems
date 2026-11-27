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

"""NVIDIA native baseline for `index_copy_`.

Mirrors benchmark/test_index_copy.py (IndexCopyBenchmark, a GenericBenchmark
subclass):
    op_name  = "index_copy_"
    torch_op = torch.Tensor.index_copy_
    dtypes   = consts.FLOAT_DTYPES

IndexCopyBenchmark.set_shapes filters the core list to <= 2**26 elements (the
int64 index plus randperm's sort workspace drive peak memory to ~24x the input)
and then appends (1, 2), (4096, 256) and (200, 40999, 3) unconditionally, so
those three are part of the core grid as well. The filter drops the two
2**30-element "from perf" shapes.

_tensor_input_fn: dim is 0 for a 1-D input and 1 otherwise; the index is
randperm(index_len) where index_len = dim_size // 2 (or 1 when dim_size < 2),
and src matches the input shape with that dim replaced by index_len.

Note the benchmark passes `inplace=True`, not `is_inplace=True`.
Benchmark.__init__ only forwards a kwarg whose attribute already exists and
`inplace` is not one, so that flag is dropped and the benchmark treats this
in-place operator as out-of-place. IS_INPLACE is True here so the collector
restores the destination outside every timed interval. See Q&A.md.
"""

import torch

OP_NAME = "index_copy_"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)
NATIVE_UNAVAILABLE_REASON = (
    "torch.Tensor.index_copy_ is unavailable on this torch build"
)

# yaml:Benchmark filtered to <= MAX_ELEMENTS (2**26), plus the three shapes
# IndexCopyBenchmark.set_shapes appends unconditionally.
_MAX_ELEMENTS = 2**26
_SHAPES = [
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1, 2),
    (4096, 256),
    (200, 40999, 3),
]

_INDEX_SEED = 0


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
    op = _resolve("torch.Tensor.index_copy_")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = 0 if len(shape) == 1 else 1
    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    return (inp, dim, index, src), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dim = 0 if len(shape) == 1 else 1
    index_max = shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    src_shape = list(shape)
    src_shape[dim] = index_len
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "written in place"},
                {"scalar": dim},
                {"shape": [index_len], "dtype": "torch.int64"},
                {"shape": src_shape, "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"self": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "dim": dim, "index_len": index_len},
    }
