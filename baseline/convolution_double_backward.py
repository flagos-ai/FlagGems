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

"""NV baseline for ``convolution_double_backward``.

Native: ``aten::_convolution_double_backward``

Grid: 10 collection point(s) over 2 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import _common
import torch

OP_NAME = "convolution_double_backward"
DTYPES = [torch.float32, torch.float16]
IS_INPLACE = False
ATEN_NAME = "_convolution_double_backward"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [
            ["T", [16, 32, 64, 64]],
            ["T", [64, 32, 3, 3]],
            ["T", [64]],
            ["T", [16, 64, 64, 64]],
            ["T", [64, 32, 3, 3]],
            ["T", [16, 32, 64, 64]],
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [16, 32, 64, 64],
    },
    {
        "args": [
            ["T", [8, 64, 32, 32]],
            ["T", [128, 64, 3, 3]],
            ["T", [128]],
            ["T", [8, 128, 16, 16]],
            ["T", [128, 64, 3, 3]],
            ["T", [8, 64, 32, 32]],
            [2, 2],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [8, 64, 32, 32],
    },
    {
        "args": [
            ["T", [8, 32, 32, 32]],
            ["T", [32, 16, 3, 3]],
            ["T", [32]],
            ["T", [8, 32, 32, 32]],
            ["T", [32, 16, 3, 3]],
            ["T", [8, 32, 32, 32]],
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            2,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [8, 32, 32, 32],
    },
    {
        "args": [
            ["T", [8, 64, 16, 16]],
            ["T", [64, 32, 3, 3]],
            ["T", [32]],
            ["T", [8, 32, 31, 31]],
            ["T", [64, 32, 3, 3]],
            ["T", [8, 64, 16, 16]],
            [2, 2],
            [1, 1],
            [1, 1],
            True,
            [0, 0],
            1,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [8, 64, 16, 16],
    },
    {
        "args": [
            ["T", [8, 32, 16, 16]],
            ["T", [32, 16, 3, 3]],
            ["T", [32]],
            ["T", [8, 32, 31, 31]],
            ["T", [32, 16, 3, 3]],
            ["T", [8, 32, 16, 16]],
            [2, 2],
            [1, 1],
            [1, 1],
            True,
            [0, 0],
            2,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [8, 32, 16, 16],
    },
    {
        "args": [
            ["T", [16, 32, 1024]],
            ["T", [64, 32, 3]],
            ["T", [64]],
            ["T", [16, 64, 1024]],
            ["T", [64, 32, 3]],
            ["T", [16, 32, 1024]],
            [1],
            [1],
            [1],
            False,
            [0],
            1,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [16, 32, 1024],
    },
    {
        "args": [
            ["T", [8, 64, 512]],
            ["T", [128, 64, 3]],
            ["T", [128]],
            ["T", [8, 128, 256]],
            ["T", [128, 64, 3]],
            ["T", [8, 64, 512]],
            [2],
            [1],
            [1],
            False,
            [0],
            1,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [8, 64, 512],
    },
    {
        "args": [
            ["T", [8, 32, 512]],
            ["T", [32, 16, 3]],
            ["T", [32]],
            ["T", [8, 32, 512]],
            ["T", [32, 16, 3]],
            ["T", [8, 32, 512]],
            [1],
            [1],
            [1],
            False,
            [0],
            2,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [8, 32, 512],
    },
    {
        "args": [
            ["T", [8, 64, 256]],
            ["T", [64, 32, 3]],
            ["T", [32]],
            ["T", [8, 32, 511]],
            ["T", [64, 32, 3]],
            ["T", [8, 64, 256]],
            [2],
            [1],
            [1],
            True,
            [0],
            1,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [8, 64, 256],
    },
    {
        "args": [
            ["T", [8, 32, 256]],
            ["T", [32, 16, 3]],
            ["T", [32]],
            ["T", [8, 32, 511]],
            ["T", [32, 16, 3]],
            ["T", [8, 32, 256]],
            [2],
            [1],
            [1],
            True,
            [0],
            2,
            [True, True, True],
        ],
        "kwargs": {},
        "key": [8, 32, 256],
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
