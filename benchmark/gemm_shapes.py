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

"""Shared, tensor-free shape and layout definitions for GEMM validation."""

from functools import lru_cache
from pathlib import Path

import yaml

GEMM_OPS = frozenset(("mm", "addmm", "bmm", "baddbmm", "router_gemm"))
CORE_SHAPES = Path(__file__).with_name("core_shapes.yaml")


@lru_cache(maxsize=1)
def _catalog():
    with CORE_SHAPES.open() as stream:
        return yaml.safe_load(stream)


def accuracy_shapes(op, quick=False):
    entry = _catalog()[op]
    key = "quick_shapes" if quick else "accuracy_shapes"
    return [tuple(shape) for shape in entry[key]]


def gemm_layouts(op, shape, requested=None, comprehensive=False):
    """True means transposed-contiguous B, or a column-major MV matrix."""
    if op not in GEMM_OPS and op != "mv":
        return (False, True) if comprehensive else (False,)
    if requested == "nn":
        return (False,)
    if requested == "nt":
        return (True,)
    equivalent = min(shape) == 1 if op == "mv" else min(shape[-2:]) == 1
    return (False,) if equivalent else (False, True)


def blas_cases(op, shapes, requested=None, comprehensive=False):
    if op in GEMM_OPS:
        for shape in shapes:
            for column_major in gemm_layouts(op, shape, requested, comprehensive):
                yield (*shape, column_major)
    else:
        for column_major in (False, True) if comprehensive else (False,):
            for shape in shapes:
                yield (*shape, column_major)


def accuracy_cases(op, quick=False):
    for shape in accuracy_shapes(op, quick):
        for column_major in gemm_layouts(op, shape):
            yield (*shape, column_major)


def input_metadata(op, args):
    if op not in GEMM_OPS and op != "mv":
        return None
    tensors = [value for value in args if hasattr(value, "stride")]
    if op in ("addmm", "baddbmm"):
        a, b = tensors[1:3]
    else:
        a, b = tensors[:2]
    if op == "mv":
        m, k = a.shape
        shape = {"B": 1, "M": m, "N": 1, "K": k}
        layout = (
            "equivalent"
            if min(a.shape) == 1
            else "row-major" if a.stride(-1) == 1 else "column-major"
        )
    else:
        if op == "router_gemm":
            b = b.t()
        m, k = a.shape[-2:]
        n = b.shape[-1]
        shape = {"B": a.shape[0] if a.ndim == 3 else 1, "M": m, "N": n, "K": k}
        layout = (
            "equivalent" if min(k, n) == 1 else ("NN" if b.stride(-1) == 1 else "NT")
        )
    return {
        "shape": shape,
        "layout": layout,
        "inputs": [
            {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)}
            for t in tensors
        ],
        "output_dtype": "torch.float32" if op == "router_gemm" else str(a.dtype),
    }
