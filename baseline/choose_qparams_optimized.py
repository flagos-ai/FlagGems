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

"""NVIDIA native baseline for `choose_qparams_optimized`.

Mirrors benchmark/test_choose_qparams_optimized.py (ChooseQParamsBenchmark, a
GenericBenchmark subclass):
    op_name  = "choose_qparams_optimized"
    torch_op = torch_choose_qparams_optimized (the module-local reference)
    dtypes   = [torch.float32]

The class overrides set_shapes with a fixed [(256,), (1024,), (4096,)] list and
returns [] from set_more_shapes, so core_shapes.yaml is not consulted: the ATen
reference is an iterative CPU-only kernel and large shapes would make it
intractable.

CAUTION: the reference this baseline measures runs on the CPU. The benchmark's
torch_op moves the input with `inp.cpu()` before calling
torch.choose_qparams_optimized, because ATen implements this operator for CPU
only. The recorded latency is therefore a host-side number for a device-side
tensor copy plus a CPU kernel, and NCU will find no device kernel to profile.
See Q&A.md.

The call is (input, numel, n_bins=200, ratio=0.16, bit_width=8) and the input
is flattened with .view(-1).
"""

import torch

OP_NAME = "choose_qparams_optimized"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.choose_qparams_optimized is unavailable on this torch build"
)

# ChooseQParamsBenchmark.set_shapes -- hand-picked for a CPU-only reference
_SHAPES = [
    (256,),
    (1024,),
    (4096,),
]

_N_BINS = 200
_RATIO = 0.16
_BIT_WIDTH = 8


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


def _torch_choose_qparams_optimized(inp, numel, n_bins, ratio, bit_width):
    """benchmark/test_choose_qparams_optimized.py::torch_choose_qparams_optimized.

    The aten reference only has a CPU implementation, so the input is moved to
    the host first -- exactly as the benchmark reference does.
    """
    return torch.choose_qparams_optimized(inp.cpu(), numel, n_bins, ratio, bit_width)


def native():
    if _resolve("torch.choose_qparams_optimized") is None:
        return None
    return _torch_choose_qparams_optimized


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    numel = 1
    for s in shape:
        numel *= s
    inp = torch.randn(shape, dtype=dtype, device=device).view(-1)
    return (inp, numel, _N_BINS, _RATIO, _BIT_WIDTH), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    numel = 1
    for s in shape:
        numel *= s
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [numel], "dtype": dt},
                {"scalar": numel},
                {"scalar": _N_BINS},
                {"scalar": _RATIO},
                {"scalar": _BIT_WIDTH},
            ],
            "kwargs": {},
        },
        "outputs": {
            "scale": {"shape": [], "dtype": "torch.float64"},
            "zero_point": {"shape": [], "dtype": "torch.float64"},
        },
        "dims": {
            "shape": shape,
            "numel": numel,
            "n_bins": _N_BINS,
            "ratio": _RATIO,
            "bit_width": _BIT_WIDTH,
        },
    }
