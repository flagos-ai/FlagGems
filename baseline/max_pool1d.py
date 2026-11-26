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

"""NV baseline for ``max_pool1d``.

Native: ``aten::max_pool1d``

Grid: 5 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import _common
import torch

OP_NAME = "max_pool1d"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "max_pool1d"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [["T", [4, 3, 224]]],
        "kwargs": {
            "kernel_size": [3],
            "stride": [2],
            "padding": [1],
            "dilation": [1],
            "ceil_mode": False,
        },
        "key": [4, 3, 224],
    },
    {
        "args": [["T", [16, 64, 128]]],
        "kwargs": {
            "kernel_size": [3],
            "stride": [2],
            "padding": [1],
            "dilation": [1],
            "ceil_mode": False,
        },
        "key": [16, 64, 128],
    },
    {
        "args": [["T", [32, 128, 64]]],
        "kwargs": {
            "kernel_size": [3],
            "stride": [2],
            "padding": [1],
            "dilation": [1],
            "ceil_mode": False,
        },
        "key": [32, 128, 64],
    },
    {
        "args": [["T", [64, 256, 32]]],
        "kwargs": {
            "kernel_size": [3],
            "stride": [2],
            "padding": [1],
            "dilation": [1],
            "ceil_mode": False,
        },
        "key": [64, 256, 32],
    },
    {
        "args": [["T", [128, 512, 16]]],
        "kwargs": {
            "kernel_size": [3],
            "stride": [2],
            "padding": [1],
            "dilation": [1],
            "ceil_mode": False,
        },
        "key": [128, 512, 16],
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
