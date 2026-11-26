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

"""NV baseline for ``max_pool3d_with_indices_backward``.

Native: ``aten::max_pool3d_with_indices_backward``

Grid: 6 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import torch

import _common

OP_NAME = "max_pool3d_with_indices_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "max_pool3d_with_indices_backward"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [
            ['T', [4, 3, 8, 28, 28]],
            ['T', [4, 3, 16, 56, 56]],
            3,
            2,
            1,
            1,
            False,
            ['T:torch.int64!index', [4, 3, 8, 28, 28]],
        ],
        "kwargs": {},
        "key": [4, 3, 8, 28, 28],
    },
    {
        "args": [
            ['T', [4, 3, 15, 28, 28]],
            ['T', [4, 3, 16, 56, 56]],
            [2, 3, 3],
            [1, 2, 2],
            [0, 1, 1],
            1,
            False,
            ['T:torch.int64!index', [4, 3, 15, 28, 28]],
        ],
        "kwargs": {},
        "key": [4, 3, 15, 28, 28],
    },
    {
        "args": [
            ['T', [8, 64, 4, 14, 14]],
            ['T', [8, 64, 8, 28, 28]],
            3,
            2,
            1,
            1,
            False,
            ['T:torch.int64!index', [8, 64, 4, 14, 14]],
        ],
        "kwargs": {},
        "key": [8, 64, 4, 14, 14],
    },
    {
        "args": [
            ['T', [8, 64, 7, 14, 14]],
            ['T', [8, 64, 8, 28, 28]],
            [2, 3, 3],
            [1, 2, 2],
            [0, 1, 1],
            1,
            False,
            ['T:torch.int64!index', [8, 64, 7, 14, 14]],
        ],
        "kwargs": {},
        "key": [8, 64, 7, 14, 14],
    },
    {
        "args": [
            ['T', [16, 128, 2, 7, 7]],
            ['T', [16, 128, 4, 14, 14]],
            3,
            2,
            1,
            1,
            False,
            ['T:torch.int64!index', [16, 128, 2, 7, 7]],
        ],
        "kwargs": {},
        "key": [16, 128, 2, 7, 7],
    },
    {
        "args": [
            ['T', [32, 256, 1, 4, 4]],
            ['T', [32, 256, 2, 7, 7]],
            3,
            2,
            1,
            1,
            False,
            ['T:torch.int64!index', [32, 256, 1, 4, 4]],
        ],
        "kwargs": {},
        "key": [32, 256, 1, 4, 4],
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
