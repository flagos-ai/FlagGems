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

"""NVIDIA native baseline for `to_sparse_semi_structured`.

Mirrors benchmark/test_to_sparse_semi_structured.py
(ToSparseSemiStructuredBenchmark, a base.Benchmark subclass):
    op_name  = "to_sparse_semi_structured"
    torch_op = torch.ops.aten._to_sparse_semi_structured
    dtypes   = [float16, bfloat16, float32, int8]

The class overrides get_input_iter with its own four square shapes and ignores
self.shapes, so core_shapes.yaml is not consulted.

The input must already satisfy the 2:4 (or 1:2 for float32) sparsity pattern,
so `_valid_dense` builds it explicitly: within each group of KSPARSE columns
exactly KSPARSE/2 slots are nonzero, the kept slots rotate with the group index,
and the float values are pushed to magnitude >= 1 so a retained slot can never
read as zero in the kernel's validity check. That construction is reproduced
verbatim.
"""

import torch

OP_NAME = "to_sparse_semi_structured"
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.int8]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._to_sparse_semi_structured is unavailable on this build"
)

# ToSparseSemiStructuredBenchmark.get_input_iter
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (2048, 2048),
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


def _valid_dense(shape, dtype, device):
    """benchmark::_valid_dense -- a dense tensor already in 2:4 (or 1:2) form."""
    rows, cols = shape
    ksparse = 2 if dtype == torch.float32 else 4
    keep = ksparse // 2
    x = torch.zeros(shape, dtype=dtype, device=device)
    for g0 in range(cols // ksparse):
        base = g0 * ksparse
        slots = [(g0 + i) % ksparse for i in range(keep)]
        if dtype.is_floating_point:
            vals = torch.randn(rows, keep, dtype=dtype, device=device)
            # Magnitude >= 1 so a retained slot never reads as zero.
            vals = torch.where(vals >= 0, vals + 1.0, vals - 1.0)
        else:
            vals = torch.randint(1, 128, (rows, keep), device=device).to(dtype)
        x[:, base + slots[0]] = vals[:, 0]
        if keep == 2:
            x[:, base + slots[1]] = vals[:, 1]
    return x


def native():
    op = _resolve("torch.ops.aten._to_sparse_semi_structured")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    return (_valid_dense(tuple(binding["shape"]), dtype, device),), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    ksparse = 2 if dtype == torch.float32 else 4
    return {
        "inputs": {
            "args": [
                {
                    "shape": shape,
                    "dtype": dt,
                    "note": f"{ksparse // 2}:{ksparse} sparse",
                }
            ],
            "kwargs": {},
        },
        "outputs": {
            "compressed": {"dtype": dt},
            "meta": {"note": "sparse metadata"},
        },
        "dims": {"M": shape[0], "N": shape[1], "ksparse": ksparse},
    }
