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

"""NV baseline for ``unsafe_index``.

Native: ``aten::_unsafe_index.Tensor``

Grid: 7 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import _common
import torch

OP_NAME = "unsafe_index"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "_unsafe_index.Tensor"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [["T", [268435456]], [["T:torch.int64!index", [65536]]]],
        "kwargs": {},
        "key": [268435456],
    },
    {
        "args": [
            ["T", [32, 32]],
            [["T:torch.int64!index", [8]], ["T:torch.int64!index", [8]]],
        ],
        "kwargs": {},
        "key": [32, 32],
    },
    {
        "args": [
            ["T", [512, 512, 512]],
            [
                ["T:torch.int64!index", [2, 128]],
                ["T:torch.int64!index", [2, 128]],
                ["T:torch.int64!index", [2, 128]],
            ],
        ],
        "kwargs": {},
        "key": [512, 512, 512],
    },
    {
        "args": [
            ["T", [64, 64, 64]],
            [["T:torch.int64!index", [2, 8]], ["T:torch.int64!index", [2, 8]]],
        ],
        "kwargs": {},
        "key": [64, 64, 64],
    },
    {
        "args": [["T", [50257, 4096]], [["T:torch.int64!index", [2048]]]],
        "kwargs": {},
        "key": [50257, 4096],
    },
    {
        "args": [
            ["T", [64, 64]],
            [["T:torch.int64!index", [2, 3, 4]], ["T:torch.int64!index", [2, 3, 4]]],
        ],
        "kwargs": {},
        "key": [64, 64],
    },
    {
        "args": [
            ["T", [4, 5, 6, 7, 8]],
            [
                ["T:torch.int64!index", [2]],
                ["T:torch.int64!index", [2]],
                ["T:torch.int64!index", [2]],
                ["T:torch.int64!index", [2]],
                ["T:torch.int64!index", [2]],
            ],
        ],
        "kwargs": {},
        "key": [4, 5, 6, 7, 8],
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
