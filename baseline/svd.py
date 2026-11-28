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

"""NVIDIA native baseline for `svd`.

Mirrors benchmark/test_svd.py (SvdBenchmark, a base.Benchmark subclass):
    op_name  = "svd"
    torch_op = torch.svd
    dtypes   = not passed, so the class's DEFAULT_DTYPES = [torch.float32]
    shapes   = yaml:svd (benchmark/core_shapes.yaml),
               shape_desc "(*B), M, N"

get_input_iter yields (inp, some=True, compute_uv=True) with both flags as
keywords.
"""

import torch

OP_NAME = "svd"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.svd is unavailable on this torch build"

# yaml:svd -- shape_desc "(*B), M, N"
_SHAPES = [
    (4096, 2, 2),
    (4096, 16, 2),
    (4096, 2, 16),
    (4096, 8, 8),
    (256, 16, 8),
    (256, 8, 16),
    (64, 32),
    (32, 64),
    (16, 16, 16),
    (17, 17),
    (256, 256),
]

_SOME = True
_COMPUTE_UV = True


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
    op = _resolve("torch.svd")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (inp,), {"some": _SOME, "compute_uv": _COMPUTE_UV}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    batch, m, n = shape[:-2], shape[-2], shape[-1]
    k = min(m, n)
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {
                "some": {"scalar": _SOME},
                "compute_uv": {"scalar": _COMPUTE_UV},
            },
        },
        "outputs": {
            "U": {"shape": batch + [m, k], "dtype": dt},
            "S": {"shape": batch + [k], "dtype": dt},
            "V": {"shape": batch + [n, k], "dtype": dt},
        },
        "dims": {"shape": shape, "M": m, "N": n},
    }
