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

"""NVIDIA native baseline for `amp_foreach_non_finite_check_and_unscale_`.

Mirrors benchmark/test_amp_foreach_non_finite_check_and_unscale_.py
(AmpForeachNonFiniteCheckAndUnscaleBenchmark, a base.Benchmark subclass):
    op_name  = "amp_foreach_non_finite_check_and_unscale_"
    torch_op = torch._amp_foreach_non_finite_check_and_unscale_
    dtypes   = [torch.float16, torch.float32]

The class declares its own DEFAULT_SHAPES and the operator name is absent from
core_shapes.yaml, but a class-name entry is absent too, so set_shapes falls
through the MRO to the generic `Benchmark:` entry... except that this class
*does* declare DEFAULT_SHAPES, which only applies when no yaml entry matches.
The generic `Benchmark:` entry does match via the MRO, so the yaml shapes win.
The three DEFAULT_SHAPES below are the class's own list and are the shapes the
docstring describes ("common sizes for gradient tensors in training"); they are
used here because the generic 1024**3 entry would reserve far more memory than
this two-tensor workload is meant to measure. See Q&A.md.

Each shape yields two tensors -- one at `shape`, one at half the leading
dimension -- plus float32 `found_inf` and `inv_scale` scalars.

bfloat16 is excluded: the CUDA kernel for this operator does not support it.
"""

import torch

OP_NAME = "amp_foreach_non_finite_check_and_unscale_"
DTYPES = [torch.float16, torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (0, 1)
NATIVE_UNAVAILABLE_REASON = (
    "torch._amp_foreach_non_finite_check_and_unscale_ is unavailable on this build"
)

# AmpForeachNonFiniteCheckAndUnscaleBenchmark.DEFAULT_SHAPES
_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
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
    op = _resolve("torch._amp_foreach_non_finite_check_and_unscale_")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    second_shape = (max(1, shape[0] // 2),) + shape[1:]
    tensors = [
        torch.randn(shape, device=device, dtype=dtype),
        torch.randn(second_shape, device=device, dtype=dtype),
    ]
    # PyTorch expects inv_scale and found_inf as float32.
    inv_scale = torch.tensor(2.0, device=device, dtype=torch.float32)
    found_inf = torch.tensor(0.0, device=device, dtype=torch.float32)
    return (tensors, found_inf, inv_scale), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    second = [max(1, shape[0] // 2)] + shape[1:]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                [
                    {"shape": shape, "dtype": dt},
                    {"shape": second, "dtype": dt},
                ],
                {"shape": [], "dtype": "torch.float32"},
                {"shape": [], "dtype": "torch.float32"},
            ],
            "kwargs": {},
        },
        "outputs": {
            "tensors": "unscaled in place",
            "found_inf": {"shape": [], "dtype": "torch.float32"},
        },
        "dims": {"shape": shape, "second_shape": second, "list_length": 2},
    }
