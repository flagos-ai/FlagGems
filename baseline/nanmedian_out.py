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

"""NV baseline for ``nanmedian_out``.

Native: ``aten::nanmedian.out``

Grid: 12 collection point(s) over 7 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (N or M, N).
"""

import torch

import _common

OP_NAME = "nanmedian_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.int8, torch.int16, torch.int32, torch.uint8]
IS_INPLACE = False
ATEN_NAME = "nanmedian.out"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [64]]],
        "kwargs": {'out': ['T', []]},
        "key": [64],
    },
    {
        "args": [['T', [64, 64]]],
        "kwargs": {'out': ['T', []]},
        "key": [64, 64],
    },
    {
        "args": [['T', [1024, 1024]]],
        "kwargs": {'out': ['T', []]},
        "key": [1024, 1024],
    },
    {
        "args": [['T', [256, 4096]]],
        "kwargs": {'out': ['T', []]},
        "key": [256, 4096],
    },
    {
        "args": [['T', [16, 131072]]],
        "kwargs": {'out': ['T', []]},
        "key": [16, 131072],
    },
    {
        "args": [['T', [268435456]]],
        "kwargs": {'out': ['T', []]},
        "key": [268435456],
    },
    {
        "args": [['T', [10000, 1]]],
        "kwargs": {'out': ['T', []]},
        "key": [10000, 1],
    },
    {
        "args": [['T', [10000, 256]]],
        "kwargs": {'out': ['T', []]},
        "key": [10000, 256],
    },
    {
        "args": [['T', [10000, 65536]]],
        "kwargs": {'out': ['T', []]},
        "key": [10000, 65536],
    },
    {
        "args": [['T', [100, 1, 100]]],
        "kwargs": {'out': ['T', []]},
        "key": [100, 1, 100],
    },
    {
        "args": [['T', [100, 256, 100]]],
        "kwargs": {'out': ['T', []]},
        "key": [100, 256, 100],
    },
    {
        "args": [['T', [100, 65536, 100]]],
        "kwargs": {'out': ['T', []]},
        "key": [100, 65536, 100],
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
