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

"""NVIDIA native baseline for `as_strided_scatter`.

Mirrors benchmark/test_as_strided_scatter.py (AsStridedScatterBenchmark, a
base.Benchmark subclass):
    op_name  = "as_strided_scatter"
    torch_op = torch.ops.aten.as_strided_scatter
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own list and shape_desc
"ROWS, COLS, ROW_STRIDE, COL_STRIDE, OFFSET, STORAGE_SIZE", so
core_shapes.yaml is not consulted. The three entries cover dense,
sparse-stride and one-dimensional scatter layouts.
"""

import torch

OP_NAME = "as_strided_scatter"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.as_strided_scatter is unavailable on this torch build"
)

# AsStridedScatterBenchmark.set_shapes -- dense, sparse-stride and 1-D layouts.
# (ROWS, COLS, ROW_STRIDE, COL_STRIDE, OFFSET, STORAGE_SIZE)
_SHAPES = [
    (128, 256, 256, 1, 0, 128 * 256),
    (64, 64, 130, 2, 5, 64 * 130 + 2),
    (1, 65536, 131072, 2, 1, 2 * 65536),
]


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
    op = _resolve("torch.ops.aten.as_strided_scatter")
    return op if callable(op) else None


def grid():
    return [
        {
            "rows": rows,
            "cols": cols,
            "row_stride": row_stride,
            "col_stride": col_stride,
            "offset": offset,
            "storage_size": storage_size,
        }
        for rows, cols, row_stride, col_stride, offset, storage_size in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    rows, cols = binding["rows"], binding["cols"]
    inp = torch.randn(binding["storage_size"], dtype=dtype, device=device)
    src = torch.randn((rows, cols), dtype=dtype, device=device)
    return (
        inp,
        src,
        (rows, cols),
        (binding["row_stride"], binding["col_stride"]),
        binding["offset"],
    ), {}


def key_shape(binding):
    return (
        f"r{binding['rows']}_c{binding['cols']}"
        f"_rs{binding['row_stride']}_cs{binding['col_stride']}"
        f"_off{binding['offset']}_store{binding['storage_size']}"
    )


def config(binding, dtype):
    rows, cols = binding["rows"], binding["cols"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [binding["storage_size"]], "dtype": dt},
                {"shape": [rows, cols], "dtype": dt},
                [rows, cols],
                [binding["row_stride"], binding["col_stride"]],
                binding["offset"],
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [binding["storage_size"]], "dtype": dt}},
        "dims": {
            "ROWS": rows,
            "COLS": cols,
            "ROW_STRIDE": binding["row_stride"],
            "COL_STRIDE": binding["col_stride"],
            "OFFSET": binding["offset"],
            "STORAGE_SIZE": binding["storage_size"],
        },
    }
