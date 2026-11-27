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

"""NVIDIA native baseline for `index_fill`.

Mirrors benchmark/test_index_fill.py (IndexFillBenchmark, a GenericBenchmark
subclass):
    op_name  = "index_fill"
    torch_op = torch.index_fill
    dtypes   = [torch.float16, torch.float32, torch.bfloat16]

The class overrides set_shapes to use its own DEFAULT_SHAPES and only extends
them at --level comprehensive, so core_shapes.yaml is not consulted and the
(8192, 4096) / (200, 40999, 3) shapes are not part of this grid.

index_fill_input_fn crosses each shape with the dims it applies to -- (0,) for
a 1-D input, (0, 1) otherwise -- and with the index lengths. For dim 0 the
lengths come from DIM0_INDEX_LENGTHS when the shape has an entry there, and
otherwise from the 1/16, 1/2 and full ratios. A (dim != 0) case whose selected
element count falls below MIN_SELECTED_NUMEL (16 * 1024) is skipped. All of
that is expanded into the grid below.

The index is randperm(dim_size)[:index_len]; the fill value is 3.14159 for the
float dtypes in use.
"""

import math

import torch

OP_NAME = "index_fill"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.index_fill is unavailable on this torch build"

# IndexFillBenchmark.DEFAULT_SHAPES -- shape_desc "input shape"
_SHAPES = [
    (65536,),
    (4096, 256),
    (4096, 4096),
]

_INDEX_RATIOS = ("1/16", "1/2", "full")
_MIN_SELECTED_NUMEL = 16 * 1024
_DIM0_INDEX_LENGTHS = {
    (4096, 256): (8, 256, 2048),
    (4096, 4096): (512, 2048, 4096),
    (8192, 4096): (512, 4096, 8192),
}
_FILL_VALUE = 3.14159


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


def _dims_for_shape(shape):
    return (0,) if len(shape) == 1 else (0, 1)


def _index_len(dim_size, ratio):
    if ratio == "1/16":
        return max(1, dim_size // 16)
    if ratio == "1/2":
        return max(1, dim_size // 2)
    if ratio == "full":
        return dim_size
    raise ValueError(f"Unknown index ratio: {ratio}")


def native():
    op = _resolve("torch.index_fill")
    return op if callable(op) else None


def grid():
    bindings = []
    for shape in _SHAPES:
        for dim in _dims_for_shape(shape):
            dim_size = shape[dim]
            index_lengths = (
                _DIM0_INDEX_LENGTHS.get(tuple(shape), ()) if dim == 0 else ()
            )
            if not index_lengths:
                index_lengths = tuple(
                    _index_len(dim_size, ratio) for ratio in _INDEX_RATIOS
                )
            for index_len in dict.fromkeys(index_lengths):
                selected_numel = math.prod(shape) // dim_size * index_len
                if selected_numel < _MIN_SELECTED_NUMEL and dim != 0:
                    continue
                bindings.append(
                    {"shape": list(shape), "dim": dim, "index_len": index_len}
                )
    return bindings


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    dim = binding["dim"]
    inp = torch.randn(shape, dtype=dtype, device=device)
    index = torch.randperm(shape[dim], device=device)[: binding["index_len"]]
    return (inp, dim, index, _FILL_VALUE), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_dim{binding['dim']}_idx{binding['index_len']}"


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": binding["dim"]},
                {"shape": [binding["index_len"]], "dtype": "torch.int64"},
                {"scalar": _FILL_VALUE},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {
            "shape": shape,
            "dim": binding["dim"],
            "index_len": binding["index_len"],
            "value": _FILL_VALUE,
        },
    }
