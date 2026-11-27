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

"""NV baseline for ``cudnn_convolution``.

Native: ``aten::cudnn_convolution``

Grid: 20 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import _common
import torch

OP_NAME = "cudnn_convolution"
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
IS_INPLACE = False
ATEN_NAME = "cudnn_convolution"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [
            ["T", [4, 16, 256]],
            ["T", [16, 16, 1]],
            [0],
            [1],
            [1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [4, 16, 256],
    },
    {
        "args": [
            ["T", [16, 32, 512]],
            ["T", [32, 32, 3]],
            [2],
            [1],
            [2],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [16, 32, 512],
    },
    {
        "args": [
            ["T", [32, 64, 512]],
            ["T", [64, 64, 3]],
            [1],
            [1],
            [1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [32, 64, 512],
    },
    {
        "args": [
            ["T", [16, 64, 56, 56]],
            ["T", [128, 64, 1, 1]],
            [0, 0],
            [1, 1],
            [1, 1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [16, 64, 56, 56],
    },
    {
        "args": [
            ["T", [8, 256, 64, 64]],
            ["T", [256, 256, 3, 3]],
            [1, 1],
            [1, 1],
            [1, 1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [8, 256, 64, 64],
    },
    {
        "args": [
            ["T", [16, 32, 32, 32]],
            ["T", [32, 32, 3, 3]],
            [2, 2],
            [1, 1],
            [2, 2],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [16, 32, 32, 32],
    },
    {
        "args": [
            ["T", [2, 16, 12, 12, 12]],
            ["T", [16, 1, 3, 3, 3]],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            16,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [2, 16, 12, 12, 12],
    },
    {
        "args": [
            ["T", [2, 16, 16, 16, 16]],
            ["T", [16, 16, 3, 3, 3]],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [2, 16, 16, 16, 16],
    },
    {
        "args": [
            ["T", [2, 4, 16, 16, 16]],
            ["T", [8, 4, 3, 3, 3]],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [2, 4, 16, 16, 16],
    },
    {
        "args": [
            ["T", [16, 32, 1024]],
            ["T", [32, 1, 3]],
            [1],
            [1],
            [1],
            32,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [16, 32, 1024],
    },
    {
        "args": [
            ["T", [8, 4, 512]],
            ["T", [8, 4, 3]],
            [1],
            [1],
            [1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [8, 4, 512],
    },
    {
        "args": [
            ["T", [16, 24, 2048]],
            ["T", [96, 12, 7]],
            [3],
            [1],
            [1],
            2,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [16, 24, 2048],
    },
    {
        "args": [
            ["T", [8, 8, 8192]],
            ["T", [16, 8, 11]],
            [5],
            [4],
            [1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [8, 8, 8192],
    },
    {
        "args": [
            ["T", [64, 48, 1024]],
            ["T", [128, 48, 5]],
            [2],
            [2],
            [1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [64, 48, 1024],
    },
    {
        "args": [
            ["T", [32, 64, 128, 128]],
            ["T", [32, 64, 3, 3]],
            [2, 2],
            [1, 1],
            [1, 1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [32, 64, 128, 128],
    },
    {
        "args": [
            ["T", [16, 32, 56, 56]],
            ["T", [32, 1, 3, 3]],
            [1, 1],
            [1, 1],
            [1, 1],
            32,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [16, 32, 56, 56],
    },
    {
        "args": [
            ["T", [32, 64, 210, 210]],
            ["T", [16, 64, 5, 5]],
            [1, 1],
            [2, 2],
            [1, 1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [32, 64, 210, 210],
    },
    {
        "args": [
            ["T", [16, 32, 24, 24]],
            ["T", [24, 16, 3, 3]],
            [2, 2],
            [2, 2],
            [1, 1],
            2,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [16, 32, 24, 24],
    },
    {
        "args": [
            ["T", [8, 3, 224, 224]],
            ["T", [16, 3, 3, 3]],
            [1, 1],
            [1, 1],
            [1, 1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [8, 3, 224, 224],
    },
    {
        "args": [
            ["T", [4, 16, 24, 24, 24]],
            ["T", [16, 16, 3, 3, 3]],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            1,
            False,
            False,
            False,
        ],
        "kwargs": {},
        "key": [4, 16, 24, 24, 24],
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
