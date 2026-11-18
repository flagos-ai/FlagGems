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

"""NVIDIA baseline for nonzero_static (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.nonzero_static``; when the installed torch lacks it, the module raises
instead of collecting a baseline from a torch composition. Shapes, dtypes,
sparsity ratios and ``size``/``fill_value`` are copied from
``benchmark/test_nonzero_static.py``.
"""

import torch

OP_NAME = "nonzero_static"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/test_nonzero_static.py::BENCH_CASES
_CASES = [
    ((1024,), 0.0, 128, -1),
    ((1024,), 0.1, 128, -1),
    ((16384,), 0.001, 128, -1),
    ((16384,), 0.1, 1024, -1),
    ((262144,), 0.001, 1024, -1),
    ((262144,), 0.1, 4096, -1),
    ((1048576,), 0.001, 1024, -1),
    ((1048576,), 0.1, 4096, -1),
    ((32, 1024), 0.1, 1024, -1),
    ((128, 4096), 0.01, 4096, -1),
    ((1024, 4096), 0.001, 1024, -1),
    ((1024, 4096), 0.1, 4096, -1),
    ((16, 64, 64), 0.1, 1024, -1),
    ((32, 128, 128), 0.01, 4096, -1),
]


def native():
    # Older torch builds (e.g. CUDA <= 11.4 era) may not ship this ATen op.
    return getattr(torch, "nonzero_static", None)


def grid():
    return [
        {
            "shape": shape,
            "nnz_ratio": nnz_ratio,
            "size": size,
            "fill_value": fill_value,
        }
        for shape, nnz_ratio, size, fill_value in _CASES
    ]


def build_inputs(binding, dtype, device):
    # Mirrors benchmark/test_nonzero_static.py::_make_input.
    torch.manual_seed(0)
    shape = binding["shape"]
    mask = torch.rand(shape, device=device) < binding["nnz_ratio"]
    x = torch.zeros(shape, device=device, dtype=dtype)
    x[mask] = 1.0
    return (x,), {
        "size": binding["size"],
        "fill_value": binding["fill_value"],
    }


def key_shape(binding):
    # Cases sharing a shape are told apart by sparsity and `size`; fill_value
    # is constant (-1) across _CASES and needs no suffix.
    shape = "x".join(str(dim) for dim in binding["shape"])
    return f"{shape}_nnz{binding['nnz_ratio']}_size{binding['size']}"


def config(binding, dtype):
    shape = list(binding["shape"])
    return {
        "inputs": {
            "input": {"shape": shape, "dtype": str(dtype)},
            "size": {"scalar": binding["size"]},
            "fill_value": {"scalar": binding["fill_value"]},
        },
        "outputs": {
            "out": {"shape": [binding["size"], len(shape)], "dtype": "torch.int64"},
        },
        "dims": {
            "nnz_ratio": binding["nnz_ratio"],
            "size": binding["size"],
        },
    }
