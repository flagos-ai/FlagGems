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

"""NVIDIA native baseline for `special_logsumexp`.

Mirrors benchmark/test_special_logsumexp.py (LogsumexpBenchmark, a
base.Benchmark subclass):
    op_name  = "special_logsumexp"
    torch_op = torch.special.logsumexp
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with LOGSUMEXP_SHAPES, so core_shapes.yaml is
not consulted. get_input_iter yields (inp, 1) -- reduce over dim 1.
"""

import torch

OP_NAME = "special_logsumexp"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.special.logsumexp is unavailable on this torch build"

# test_special_logsumexp.py::LOGSUMEXP_SHAPES
_SHAPES = [
    (256, 256),
    (1024, 1024),
    (4096, 4096),
]

_DIM = 1


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
    op = _resolve("torch.special.logsumexp")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (inp, _DIM), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    out_shape = [d for i, d in enumerate(shape) if i != _DIM]
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": _DIM},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {"shape": shape, "dim": _DIM},
    }
