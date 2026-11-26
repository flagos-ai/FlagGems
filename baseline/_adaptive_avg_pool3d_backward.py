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

"""NV baseline for ``_adaptive_avg_pool3d_backward``.

Native: ``aten::_adaptive_avg_pool3d_backward``

Grid: 4 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import torch

import _common

OP_NAME = "_adaptive_avg_pool3d_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "_adaptive_avg_pool3d_backward"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [1, 3, 4, 4, 4]], ['T', [1, 3, 8, 8, 8]]],
        "kwargs": {},
        "key": [1, 3, 4, 4, 4],
    },
    {
        "args": [['T', [2, 3, 8, 8, 8]], ['T', [2, 3, 16, 16, 16]]],
        "kwargs": {},
        "key": [2, 3, 8, 8, 8],
    },
    {
        "args": [['T', [1, 1, 16, 16, 16]], ['T', [1, 1, 32, 32, 32]]],
        "kwargs": {},
        "key": [1, 1, 16, 16, 16],
    },
    {
        "args": [['T', [4, 8, 32, 32, 32]], ['T', [4, 8, 64, 64, 64]]],
        "kwargs": {},
        "key": [4, 8, 32, 32, 32],
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
