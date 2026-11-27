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

"""NVIDIA native baseline for `geqrf`.

Mirrors benchmark/test_geqrf.py (GeqrfBenchmark, a base.Benchmark subclass):
    op_name  = "geqrf"
    torch_op = torch.ops.aten.geqrf
    dtypes   = [torch.float32]

The class overrides set_shapes with GEQRF_SHAPES and shape_desc
"N, N | M, N | batch, N, N", so core_shapes.yaml is not consulted.

The input is conditioned the way the benchmark conditions it:
A = randn(shape) + m * eye(m, n), a diagonal bump that keeps QR away from
singularity (the same recipe the det/slogdet/solve_ex benchmarks use).

float64 is in the dtype list because the benchmark appends it when
flag_gems.runtime.device.support_fp64 is true, and the NVIDIA backend
reports fp64_enabled=True.
"""

import torch

OP_NAME = "geqrf"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.ops.aten.geqrf is unavailable on this torch build"

# test_geqrf.py::GEQRF_SHAPES -- "N, N | M, N | batch, N, N"
_SHAPES = [
    (64, 64),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (1024, 128),
    (8, 64, 64),
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
    op = _resolve("torch.ops.aten.geqrf")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    m, n = shape[-2], shape[-1]
    a = torch.randn(shape, dtype=dtype, device=device)
    a = a + m * torch.eye(m, n, dtype=dtype, device=device)
    return (a,), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    m, n = shape[-2], shape[-1]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {},
        },
        "outputs": {
            "a": {"shape": shape, "dtype": dt},
            "tau": {"shape": shape[:-2] + [min(m, n)], "dtype": dt},
        },
        "dims": {"shape": shape, "M": m, "N": n},
    }
