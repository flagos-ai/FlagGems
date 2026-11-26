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

"""NV baseline for ``upsample_bilinear2d_aa_backward``.

Native: ``aten::_upsample_bilinear2d_aa_backward``

Grid: 16 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import torch

import _common

OP_NAME = "upsample_bilinear2d_aa_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "_upsample_bilinear2d_aa_backward"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [4, 16, 1, 1]], [1, 1], [4, 16, 4, 4], False, None, None],
        "kwargs": {},
        "key": [4, 16, 1, 1],
    },
    {
        "args": [['T', [4, 16, 16, 16]], [16, 16], [4, 16, 4, 4], False, None, None],
        "kwargs": {},
        "key": [4, 16, 16, 16],
    },
    {
        "args": [['T', [4, 16, 4, 4]], [4, 4], [4, 16, 16, 16], False, None, None],
        "kwargs": {},
        "key": [4, 16, 4, 4],
    },
    {
        "args": [['T', [4, 16, 64, 128]], [64, 128], [4, 16, 16, 32], False, None, None],
        "kwargs": {},
        "key": [4, 16, 64, 128],
    },
    {
        "args": [['T', [1, 1, 16, 16]], [16, 16], [1, 1, 64, 64], False, None, None],
        "kwargs": {},
        "key": [1, 1, 16, 16],
    },
    {
        "args": [['T', [1, 1, 32, 32]], [32, 32], [1, 1, 64, 64], False, None, None],
        "kwargs": {},
        "key": [1, 1, 32, 32],
    },
    {
        "args": [['T', [1, 1, 128, 128]], [128, 128], [1, 1, 64, 64], False, None, None],
        "kwargs": {},
        "key": [1, 1, 128, 128],
    },
    {
        "args": [['T', [4, 3, 128, 128]], [128, 128], [4, 3, 256, 256], False, None, None],
        "kwargs": {},
        "key": [4, 3, 128, 128],
    },
    {
        "args": [['T', [4, 3, 256, 256]], [256, 256], [4, 3, 128, 128], False, None, None],
        "kwargs": {},
        "key": [4, 3, 256, 256],
    },
    {
        "args": [['T', [4, 64, 32, 32]], [32, 32], [4, 64, 64, 64], False, None, None],
        "kwargs": {},
        "key": [4, 64, 32, 32],
    },
    {
        "args": [['T', [1, 64, 128, 128]], [128, 128], [1, 64, 512, 512], False, None, None],
        "kwargs": {},
        "key": [1, 64, 128, 128],
    },
    {
        "args": [['T', [1, 64, 1024, 1024]], [1024, 1024], [1, 64, 512, 512], False, None, None],
        "kwargs": {},
        "key": [1, 64, 1024, 1024],
    },
    {
        "args": [['T', [512, 1024, 8, 8]], [8, 8], [512, 1024, 32, 32], False, None, None],
        "kwargs": {},
        "key": [512, 1024, 8, 8],
    },
    {
        "args": [['T', [256, 512, 16, 16]], [16, 16], [256, 512, 64, 64], False, None, None],
        "kwargs": {},
        "key": [256, 512, 16, 16],
    },
    {
        "args": [['T', [256, 512, 32, 32]], [32, 32], [256, 512, 64, 64], False, None, None],
        "kwargs": {},
        "key": [256, 512, 32, 32],
    },
    {
        "args": [['T', [256, 512, 128, 128]], [128, 128], [256, 512, 64, 64], False, None, None],
        "kwargs": {},
        "key": [256, 512, 128, 128],
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
