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

"""NV baseline for ``floor_divide_tensor``.

Native: ``torch.floor_divide``

Grid: 15 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import _common
import torch

OP_NAME = "floor_divide_tensor"
DTYPES = [torch.float32, torch.int16, torch.int32]
IS_INPLACE = False
ATEN_NAME = "floor_divide"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [["T", [1073741824]], ["T", [1073741824]]],
        "kwargs": {},
        "key": [1073741824],
    },
    {
        "args": [["T", [64, 64]], ["T", [64, 64]]],
        "kwargs": {},
        "key": [64, 64],
    },
    {
        "args": [["T", [4096, 4096]], ["T", [4096, 4096]]],
        "kwargs": {},
        "key": [4096, 4096],
    },
    {
        "args": [["T", [64, 512, 512]], ["T", [64, 512, 512]]],
        "kwargs": {},
        "key": [64, 512, 512],
    },
    {
        "args": [["T", [1024, 1024, 1024]], ["T", [1024, 1024, 1024]]],
        "kwargs": {},
        "key": [1024, 1024, 1024],
    },
    {
        "args": [["T", [1024, 1]], ["T", [1024, 1]]],
        "kwargs": {},
        "key": [1024, 1],
    },
    {
        "args": [["T", [1024, 16]], ["T", [1024, 16]]],
        "kwargs": {},
        "key": [1024, 16],
    },
    {
        "args": [["T", [1024, 256]], ["T", [1024, 256]]],
        "kwargs": {},
        "key": [1024, 256],
    },
    {
        "args": [["T", [1024, 4096]], ["T", [1024, 4096]]],
        "kwargs": {},
        "key": [1024, 4096],
    },
    {
        "args": [["T", [1024, 65536]], ["T", [1024, 65536]]],
        "kwargs": {},
        "key": [1024, 65536],
    },
    {
        "args": [["T", [64, 64, 1]], ["T", [64, 64, 1]]],
        "kwargs": {},
        "key": [64, 64, 1],
    },
    {
        "args": [["T", [64, 64, 16]], ["T", [64, 64, 16]]],
        "kwargs": {},
        "key": [64, 64, 16],
    },
    {
        "args": [["T", [64, 64, 256]], ["T", [64, 64, 256]]],
        "kwargs": {},
        "key": [64, 64, 256],
    },
    {
        "args": [["T", [64, 64, 4096]], ["T", [64, 64, 4096]]],
        "kwargs": {},
        "key": [64, 64, 4096],
    },
    {
        "args": [["T", [64, 64, 65536]], ["T", [64, 64, 65536]]],
        "kwargs": {},
        "key": [64, 64, 65536],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch.floor_divide``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch.floor_divide")
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
