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

"""NVIDIA native baseline for `constant_pad_nd`.

Mirrors benchmark/test_constant_pad_nd.py (base.GenericBenchmark):
    op_name  = "constant_pad_nd"
    torch_op = torch.constant_pad_nd
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

The benchmark's _input_fn draws the pad widths with `random.randint(0, 10)` for
each of the 2 * rank entries and seeds nothing, so the widths differ between
runs. A collector baseline has to be reproducible (build_inputs is called once
in the parent process and again in the NCU subprocess, and both must agree), so
the widths here are drawn from a module-local random.Random seeded per binding.
The distribution and the call shape are identical; only the seed is pinned.
See Q&A.md.

value is 1.5, as in the benchmark. pad/value are passed as keywords, matching
the dict the input_fn yields.
"""

import random

import torch

OP_NAME = "constant_pad_nd"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.constant_pad_nd is unavailable on this torch build"

# yaml:Benchmark -- the shared default shape list
_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

_VALUE = 1.5
# The benchmark uses the global `random` module unseeded; pin a seed so the
# collector's two build_inputs calls agree on the pad widths.
_PAD_SEED = 0


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


def _pad_for(shape):
    """2 * rank widths drawn from randint(0, 10), as in the benchmark."""
    rng = random.Random(_PAD_SEED)
    return [rng.randint(0, 10) for _ in range(len(shape) * 2)]


def native():
    op = _resolve("torch.constant_pad_nd")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, device=device, dtype=dtype)
    return (inp,), {"pad": _pad_for(shape), "value": _VALUE}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    pad = _pad_for(shape)
    dt = str(dtype)
    out_shape = [
        shape[i] + pad[2 * (len(shape) - 1 - i)] + pad[2 * (len(shape) - 1 - i) + 1]
        for i in range(len(shape))
    ]
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {"pad": pad, "value": {"scalar": _VALUE}},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {"shape": shape, "pad": pad, "value": _VALUE},
    }
