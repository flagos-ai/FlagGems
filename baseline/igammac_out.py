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

"""NVIDIA native baseline for `igammac_out`.

Mirrors benchmark/test_igammac.py (IgammacBenchmark, a GenericBenchmark
subclass):
    op_name  = "igammac_out"
    torch_op = _torch_igammac (the module-local reference below)
    dtypes   = [torch.float32]
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

`_torch_igammac` has two branches. The composed Lanczos/series fallback runs
only when flag_gems.device is neither "cuda" nor "cpu"; on NVIDIA it calls
torch.igammac(a, x, out=out) directly, which is what this baseline resolves.

The class's set_shapes drops the huge core shapes only on that same
non-CUDA path, so the full core list applies here.

float64 is in the dtype list because the benchmark appends it when
flag_gems.runtime.device.support_fp64 is true, and the NVIDIA backend reports
fp64_enabled=True. IgammacBenchmark.get_input_iter then restricts the float64
rows to shapes of at most MAX_FLOAT64_ELEMENTS (2**24) elements, because
torch's float64 igammac baseline is very slow on large tensors;
unsupported_reason() reproduces that filter so the collector records an
explicit row rather than silently measuring a shape the benchmark skips.

igammac(a, x) is defined for a > 0 and x >= 0, so `_igammac_input_out` draws
both from rand(shape) * 10 + 0.1 rather than randn.
"""

import torch

OP_NAME = "igammac_out"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = "torch.igammac is unavailable on this torch build"

# yaml:Benchmark -- the shared default shape list
_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

# IgammacBenchmark.MAX_FLOAT64_ELEMENTS
_MAX_FLOAT64_ELEMENTS = 2**24


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


def _torch_igammac(a, x, out=None):
    """benchmark::_torch_igammac, CUDA branch (flag_gems.device == "cuda")."""
    return torch.igammac(a, x) if out is None else torch.igammac(a, x, out=out)


def native():
    if _resolve("torch.igammac") is None:
        return None
    return _torch_igammac


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    a = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    x = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    out = torch.empty_like(a)
    return (a, x), {"out": out}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "a > 0"},
                {"shape": shape, "dtype": dt, "note": "x >= 0"},
            ],
            "kwargs": {"out": {"shape": shape, "dtype": dt}},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }


def unsupported_reason(binding, dtype):
    if dtype is torch.float64:
        numel = 1
        for d in binding["shape"]:
            numel *= d
        if numel > _MAX_FLOAT64_ELEMENTS:
            return (
                f"float64 is benchmarked only up to {_MAX_FLOAT64_ELEMENTS} "
                f"elements; this shape has {numel}"
            )
    return None
