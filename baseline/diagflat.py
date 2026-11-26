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

"""NV baseline for ``diagflat``.

Native: ``torch.diagflat``

Grid: 10 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (input shape, diagonal offset).
"""

import torch

import _common

OP_NAME = "diagflat"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "diagflat"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [64]], 0],
        "kwargs": {},
        "key": [64],
    },
    {
        "args": [['T', [256]], 0],
        "kwargs": {},
        "key": [256],
    },
    {
        "args": [['T', [1024]], 0],
        "kwargs": {},
        "key": [1024],
    },
    {
        "args": [['T', [2048]], 0],
        "kwargs": {},
        "key": [2048],
    },
    {
        "args": [['T', [4096]], 0],
        "kwargs": {},
        "key": [4096],
    },
    {
        "args": [['T', [32, 32]], 0],
        "kwargs": {},
        "key": [32, 32],
    },
    {
        "args": [['T', [64, 64]], 0],
        "kwargs": {},
        "key": [64, 64],
    },
    {
        "args": [['T', [16, 16, 16]], 0],
        "kwargs": {},
        "key": [16, 16, 16],
    },
    {
        "args": [['T', [8192]], 0],
        "kwargs": {},
        "key": [8192],
    },
    {
        "args": [['T', [128, 64]], 0],
        "kwargs": {},
        "key": [128, 64],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch.diagflat``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch.diagflat")
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
