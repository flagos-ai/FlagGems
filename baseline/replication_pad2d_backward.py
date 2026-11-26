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

"""NV baseline for ``replication_pad2d_backward``.

Native: ``aten::replication_pad2d_backward``

Grid: 24 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import torch

import _common

OP_NAME = "replication_pad2d_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "replication_pad2d_backward"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [1, 3, 258, 258]], ['T', [1, 3, 256, 256]], [1, 1, 1, 1]],
        "kwargs": {},
        "key": [1, 3, 258, 258],
    },
    {
        "args": [['T', [1, 3, 263, 259]], ['T', [1, 3, 256, 256]], [1, 2, 3, 4]],
        "kwargs": {},
        "key": [1, 3, 263, 259],
    },
    {
        "args": [['T', [1, 3, 259, 259]], ['T', [1, 3, 256, 256]], [3, 0, 0, 3]],
        "kwargs": {},
        "key": [1, 3, 259, 259],
    },
    {
        "args": [['T', [1, 3, 260, 260]], ['T', [1, 3, 256, 256]], [2, 2, 2, 2]],
        "kwargs": {},
        "key": [1, 3, 260, 260],
    },
    {
        "args": [['T', [1, 3, 642, 642]], ['T', [1, 3, 640, 640]], [1, 1, 1, 1]],
        "kwargs": {},
        "key": [1, 3, 642, 642],
    },
    {
        "args": [['T', [1, 3, 647, 643]], ['T', [1, 3, 640, 640]], [1, 2, 3, 4]],
        "kwargs": {},
        "key": [1, 3, 647, 643],
    },
    {
        "args": [['T', [1, 3, 643, 643]], ['T', [1, 3, 640, 640]], [3, 0, 0, 3]],
        "kwargs": {},
        "key": [1, 3, 643, 643],
    },
    {
        "args": [['T', [1, 3, 644, 644]], ['T', [1, 3, 640, 640]], [2, 2, 2, 2]],
        "kwargs": {},
        "key": [1, 3, 644, 644],
    },
    {
        "args": [['T', [1, 3, 1026, 1026]], ['T', [1, 3, 1024, 1024]], [1, 1, 1, 1]],
        "kwargs": {},
        "key": [1, 3, 1026, 1026],
    },
    {
        "args": [['T', [1, 3, 1031, 1027]], ['T', [1, 3, 1024, 1024]], [1, 2, 3, 4]],
        "kwargs": {},
        "key": [1, 3, 1031, 1027],
    },
    {
        "args": [['T', [1, 3, 1027, 1027]], ['T', [1, 3, 1024, 1024]], [3, 0, 0, 3]],
        "kwargs": {},
        "key": [1, 3, 1027, 1027],
    },
    {
        "args": [['T', [1, 3, 1028, 1028]], ['T', [1, 3, 1024, 1024]], [2, 2, 2, 2]],
        "kwargs": {},
        "key": [1, 3, 1028, 1028],
    },
    {
        "args": [['T', [1, 64, 130, 130]], ['T', [1, 64, 128, 128]], [1, 1, 1, 1]],
        "kwargs": {},
        "key": [1, 64, 130, 130],
    },
    {
        "args": [['T', [1, 64, 135, 131]], ['T', [1, 64, 128, 128]], [1, 2, 3, 4]],
        "kwargs": {},
        "key": [1, 64, 135, 131],
    },
    {
        "args": [['T', [1, 64, 131, 131]], ['T', [1, 64, 128, 128]], [3, 0, 0, 3]],
        "kwargs": {},
        "key": [1, 64, 131, 131],
    },
    {
        "args": [['T', [1, 64, 132, 132]], ['T', [1, 64, 128, 128]], [2, 2, 2, 2]],
        "kwargs": {},
        "key": [1, 64, 132, 132],
    },
    {
        "args": [['T', [1, 64, 258, 258]], ['T', [1, 64, 256, 256]], [1, 1, 1, 1]],
        "kwargs": {},
        "key": [1, 64, 258, 258],
    },
    {
        "args": [['T', [1, 64, 263, 259]], ['T', [1, 64, 256, 256]], [1, 2, 3, 4]],
        "kwargs": {},
        "key": [1, 64, 263, 259],
    },
    {
        "args": [['T', [1, 64, 259, 259]], ['T', [1, 64, 256, 256]], [3, 0, 0, 3]],
        "kwargs": {},
        "key": [1, 64, 259, 259],
    },
    {
        "args": [['T', [1, 64, 260, 260]], ['T', [1, 64, 256, 256]], [2, 2, 2, 2]],
        "kwargs": {},
        "key": [1, 64, 260, 260],
    },
    {
        "args": [['T', [1, 64, 514, 514]], ['T', [1, 64, 512, 512]], [1, 1, 1, 1]],
        "kwargs": {},
        "key": [1, 64, 514, 514],
    },
    {
        "args": [['T', [1, 64, 519, 515]], ['T', [1, 64, 512, 512]], [1, 2, 3, 4]],
        "kwargs": {},
        "key": [1, 64, 519, 515],
    },
    {
        "args": [['T', [1, 64, 515, 515]], ['T', [1, 64, 512, 512]], [3, 0, 0, 3]],
        "kwargs": {},
        "key": [1, 64, 515, 515],
    },
    {
        "args": [['T', [1, 64, 516, 516]], ['T', [1, 64, 512, 512]], [2, 2, 2, 2]],
        "kwargs": {},
        "key": [1, 64, 516, 516],
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
