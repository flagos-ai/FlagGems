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

"""NVIDIA native baseline for `linalg_svdvals`.

Mirrors benchmark/test_linalg_svdvals.py (SvdBenchmark, a
GenericBenchmark2DOnly subclass):
    op_name  = "linalg_svdvals"
    torch_op = torch.linalg.svdvals
    dtypes   = [torch.float32]

The class's set_more_shapes returns SVD_BENCHMARK_SHAPES, but set_more_shapes
only runs at --level comprehensive. At --level core the shapes come from the
GenericBenchmark2DOnly yaml entry, whose largest entry is (1024, 65536) -- an
SVD of that is far outside what this benchmark means to measure.

This baseline therefore uses SVD_BENCHMARK_SHAPES, the class's own list: they
are the shapes the author wrote for this operator. See Q&A.md.

The input is always built in float32 regardless of the dtype axis, as the
benchmark's input_fn does (SVD on CUDA is float32-only in PyTorch).
"""

import torch

OP_NAME = "linalg_svdvals"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.linalg.svdvals is unavailable on this torch build"

# test_linalg_svdvals.py::SVD_BENCHMARK_SHAPES
_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
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
    op = _resolve("torch.linalg.svdvals")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["shape"]
    # The benchmark's input_fn pins float32 regardless of the dtype axis.
    inp = torch.randn([m, n], dtype=torch.float32, device=device)
    return (inp,), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    m, n = binding["shape"]
    f32 = "torch.float32"
    return {
        "inputs": {
            "args": [{"shape": [m, n], "dtype": f32}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [min(m, n)], "dtype": f32}},
        "dims": {"M": m, "N": n},
    }
