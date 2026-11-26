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

"""NVIDIA native baseline for `multilabel_margin_loss_forward`.

Mirrors benchmark/test_multilabel_margin_loss_forward.py
(MultilabelMarginLossBenchmark, a GenericBenchmark2DOnly subclass):
    op_name  = "multilabel_margin_loss_forward"
    torch_op = torch.ops.aten.multilabel_margin_loss_forward
    dtypes   = consts.FLOAT_DTYPES (one parametrized run each)
    shapes   = yaml:multilabel_margin_loss_forward
               (benchmark/core_shapes.yaml), shape_desc "N, C"

The class returns [] from set_more_shapes because the generic comprehensive
list includes (10000, 65536), which is not meaningful for this deliberately
O(N*C^2) operator.

The target is arange(C) expanded over the rows with position
`stop = max(1, C // 2)` set to -1, which terminates the label list. reduction=1
(mean) is passed as a keyword. The duplicate-label and full-target variants are
COMPREHENSIVE-only.

The test skips a dtype when the native reference is missing or silently falls
back to the CPU (`_native_fallback_reason` probes for that). That probe is a
runtime check against the installed torch and is not reproduced here; if a
dtype falls back, the collected row for it is a CPU measurement. See Q&A.md.
"""

import torch

OP_NAME = "multilabel_margin_loss_forward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.multilabel_margin_loss_forward is unavailable on this build"
)

# yaml:multilabel_margin_loss_forward -- shape_desc "N, C"
_SHAPES = [
    (8, 128),
    (32, 256),
    (16, 512),
    (8, 1024),
    (2, 2048),
    (1, 4096),
]

_REDUCTION = 1  # mean


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
    op = _resolve("torch.ops.aten.multilabel_margin_loss_forward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    n_rows, n_classes = binding["shape"]
    inp = torch.randn((n_rows, n_classes), dtype=dtype, device=device)
    target = (
        torch.arange(n_classes, dtype=torch.int64, device=device)
        .expand(n_rows, n_classes)
        .clone()
    )
    stop = max(1, n_classes // 2)
    if stop < n_classes:
        target[:, stop] = -1
    return (inp, target), {"reduction": _REDUCTION}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": "torch.int64"},
            ],
            "kwargs": {"reduction": {"scalar": _REDUCTION}},
        },
        "outputs": {
            "output": {"shape": [], "dtype": dt},
            "is_target": {"shape": shape, "dtype": dt},
        },
        "dims": {"N": shape[0], "C": shape[1], "reduction": _REDUCTION},
    }
