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

"""NV baseline for ``compute_linear_combination_out``.

Native: ``aten::_compute_linear_combination.out``

Grid: 6 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import torch

import _common

OP_NAME = "compute_linear_combination_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "_compute_linear_combination.out"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [512, 512]], ['T', [64, 512]]],
        "kwargs": {'out': ['T', [64, 512]]},
        "key": [512, 512],
    },
    {
        "args": [['T', [1024, 1024]], ['T', [1024, 1024]]],
        "kwargs": {'out': ['T', [1024, 1024]]},
        "key": [1024, 1024],
    },
    {
        "args": [['T', [64, 64]], ['T', [64, 64]]],
        "kwargs": {'out': ['T', [64, 64]]},
        "key": [64, 64],
    },
    {
        "args": [['T', [256, 512]], ['T', [128, 256]]],
        "kwargs": {'out': ['T', [128, 512]]},
        "key": [256, 512],
    },
    {
        "args": [['T', [512, 1024]], ['T', [256, 512]]],
        "kwargs": {'out': ['T', [256, 1024]]},
        "key": [512, 1024],
    },
    {
        "args": [['T', [1024, 4096]], ['T', [8, 1024]]],
        "kwargs": {'out': ['T', [8, 4096]]},
        "key": [1024, 4096],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it."""
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve(ATEN_NAME)
    NATIVE_UNAVAILABLE_REASON = reason
    return op


def grid():
    return list(GRID)


def build_inputs(binding, dtype, device):
    args = tuple(_common.build_value(v, dtype, device) for v in binding["args"])
    kwargs = {
        key: _common.build_value(value, dtype, device)
        for key, value in binding["kwargs"].items()
    }
    return args, kwargs


def key_shape(binding):
    return binding["key"]


def config(binding, dtype):
    args, kwargs = build_inputs(binding, dtype, "meta")
    return _common.config_from_call(args, kwargs)
