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

"""NV baseline for ``triangular_solve``.

Native: ``aten::triangular_solve``

Grid: 3 collection point(s) over 1 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (A batch, B batch, n, nrhs, upper, transpose,
unitriangular).
"""

import torch

import _common

OP_NAME = "triangular_solve"
DTYPES = [torch.float32]
IS_INPLACE = False
ATEN_NAME = "triangular_solve"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [8, 1]], ['T', [8, 8]], False, False, False],
        "kwargs": {},
        "key": [8, 1],
    },
    {
        "args": [['T', [8, 8]], ['T', [8, 8]], False, False, False],
        "kwargs": {},
        "key": [8, 8],
    },
    {
        "args": [['T', [32, 1]], ['T', [32, 32]], False, False, False],
        "kwargs": {},
        "key": [32, 1],
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
