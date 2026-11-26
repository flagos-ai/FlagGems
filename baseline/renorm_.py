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

"""NV baseline for ``renorm_``.

Native: ``aten::renorm_``

Grid: 6 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import torch

import _common

OP_NAME = "renorm_"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "renorm_"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [16, 256]], 2.0, 1, 1.0],
        "kwargs": {},
        "key": [16, 256],
    },
    {
        "args": [['T', [32, 512]], 2.0, 1, 1.0],
        "kwargs": {},
        "key": [32, 512],
    },
    {
        "args": [['T', [64, 1024]], 2.0, 1, 1.0],
        "kwargs": {},
        "key": [64, 1024],
    },
    {
        "args": [['T', [128, 2048]], 2.0, 1, 1.0],
        "kwargs": {},
        "key": [128, 2048],
    },
    {
        "args": [['T', [256, 4096]], 2.0, 1, 1.0],
        "kwargs": {},
        "key": [256, 4096],
    },
    {
        "args": [['T', [512, 8192]], 2.0, 1, 1.0],
        "kwargs": {},
        "key": [512, 8192],
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
