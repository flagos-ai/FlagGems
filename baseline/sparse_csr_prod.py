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

"""NVIDIA native baseline for `sparse_csr_prod`.

Mirrors benchmark/test_sparse_csr_prod.py (SparseCsrProdBenchmark, a
base.Benchmark subclass):
    op_name  = "sparse_csr_prod"
    torch_op = torch.ops.aten._sparse_csr_prod
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with SPARSE_CSR_PROD_SHAPES and declares
shape_desc "M, N (CSR)", so core_shapes.yaml is not consulted.

The input is a CSR tensor at 30% density, built the way `_make_csr` builds it:
a dense randn masked by rand(shape) < density, then to_sparse_csr(). The call
is op(sp, [0], True, dtype=dtype) -- reduce over dim 0, keepdim True, and the
accumulate dtype passed as a keyword.
"""

import torch

OP_NAME = "sparse_csr_prod"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._sparse_csr_prod is unavailable on this torch build"
)

# test_sparse_csr_prod.py::SPARSE_CSR_PROD_SHAPES -- shape_desc "M, N (CSR)"
_SHAPES = [
    (1024, 1024),
    (4096, 512),
    (512, 4096),
]

_DENSITY = 0.3
_DIM = [0]
_KEEPDIM = True


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


def _make_csr(shape, dtype, device, density=_DENSITY):
    """benchmark::_make_csr -- masked dense, then CSR, on the target device."""
    x = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand(shape, device=device) < density
    return (x * mask).to_sparse_csr()


def native():
    op = _resolve("torch.ops.aten._sparse_csr_prod")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    sp = _make_csr(tuple(binding["shape"]), dtype, device)
    return (sp, list(_DIM), _KEEPDIM), {"dtype": dtype}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {
                    "shape": shape,
                    "dtype": dt,
                    "layout": "torch.sparse_csr",
                    "density": _DENSITY,
                },
                list(_DIM),
                {"scalar": _KEEPDIM},
            ],
            "kwargs": {"dtype": dt},
        },
        "outputs": {"out": {"shape": [1, shape[1]], "dtype": dt}},
        "dims": {
            "M": shape[0],
            "N": shape[1],
            "density": _DENSITY,
            "dim": list(_DIM),
            "keepdim": _KEEPDIM,
        },
    }
