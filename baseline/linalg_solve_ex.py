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

"""NV baseline for ``linalg_solve_ex``.

Native: ``torch._C._linalg.linalg_solve_ex``

Grid: 11 collection point(s) over 1 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import torch

import _common

OP_NAME = "linalg_solve_ex"
DTYPES = [torch.float32]
IS_INPLACE = False
ATEN_NAME = "linalg_solve_ex"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [1, 4, 4]], ['T', [1, 4, 1]]],
        "kwargs": {},
        "key": [1, 4, 4],
    },
    {
        "args": [['T', [1, 8, 8]], ['T', [1, 8, 1]]],
        "kwargs": {},
        "key": [1, 8, 8],
    },
    {
        "args": [['T', [1, 16, 16]], ['T', [1, 16, 1]]],
        "kwargs": {},
        "key": [1, 16, 16],
    },
    {
        "args": [['T', [4, 4, 4]], ['T', [4, 4, 4]]],
        "kwargs": {},
        "key": [4, 4, 4],
    },
    {
        "args": [['T', [4, 8, 8]], ['T', [4, 8, 4]]],
        "kwargs": {},
        "key": [4, 8, 8],
    },
    {
        "args": [['T', [4, 16, 16]], ['T', [4, 16, 4]]],
        "kwargs": {},
        "key": [4, 16, 16],
    },
    {
        "args": [['T', [8, 8, 8]], ['T', [8, 8, 1]]],
        "kwargs": {},
        "key": [8, 8, 8],
    },
    {
        "args": [['T', [8, 16, 16]], ['T', [8, 16, 1]]],
        "kwargs": {},
        "key": [8, 16, 16],
    },
    {
        "args": [['T', [16, 8, 8]], ['T', [16, 8, 4]]],
        "kwargs": {},
        "key": [16, 8, 8],
    },
    {
        "args": [['T', [16, 16, 16]], ['T', [16, 16, 4]]],
        "kwargs": {},
        "key": [16, 16, 16],
    },
    {
        "args": [['T', [32, 8, 8]], ['T', [32, 8, 1]]],
        "kwargs": {},
        "key": [32, 8, 8],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch._C._linalg.linalg_solve_ex``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch._C._linalg.linalg_solve_ex")
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
