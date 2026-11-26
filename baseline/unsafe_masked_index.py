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

"""NV baseline for ``unsafe_masked_index``.

Native: ``torch._unsafe_masked_index``

Grid: 10 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import _common
import torch

OP_NAME = "unsafe_masked_index"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "_unsafe_masked_index"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [
            ["T", [1024]],
            ["T:torch.bool", [1024]],
            [["T:torch.int64!index", [1024]]],
            0.0,
        ],
        "kwargs": {},
        "key": [1024],
    },
    {
        "args": [
            ["T", [2048]],
            ["T:torch.bool", [2048]],
            [["T:torch.int64!index", [2048]]],
            0.0,
        ],
        "kwargs": {},
        "key": [2048],
    },
    {
        "args": [
            ["T", [4096]],
            ["T:torch.bool", [4096]],
            [["T:torch.int64!index", [4096]]],
            0.0,
        ],
        "kwargs": {},
        "key": [4096],
    },
    {
        "args": [
            ["T", [8192]],
            ["T:torch.bool", [8192]],
            [["T:torch.int64!index", [8192]]],
            0.0,
        ],
        "kwargs": {},
        "key": [8192],
    },
    {
        "args": [
            ["T", [16384]],
            ["T:torch.bool", [16384]],
            [["T:torch.int64!index", [16384]]],
            0.0,
        ],
        "kwargs": {},
        "key": [16384],
    },
    {
        "args": [
            ["T", [32768]],
            ["T:torch.bool", [32768]],
            [["T:torch.int64!index", [32768]]],
            0.0,
        ],
        "kwargs": {},
        "key": [32768],
    },
    {
        "args": [
            ["T", [65536]],
            ["T:torch.bool", [65536]],
            [["T:torch.int64!index", [65536]]],
            0.0,
        ],
        "kwargs": {},
        "key": [65536],
    },
    {
        "args": [
            ["T", [131072]],
            ["T:torch.bool", [131072]],
            [["T:torch.int64!index", [131072]]],
            0.0,
        ],
        "kwargs": {},
        "key": [131072],
    },
    {
        "args": [
            ["T", [262144]],
            ["T:torch.bool", [262144]],
            [["T:torch.int64!index", [262144]]],
            0.0,
        ],
        "kwargs": {},
        "key": [262144],
    },
    {
        "args": [
            ["T", [1048576]],
            ["T:torch.bool", [1048576]],
            [["T:torch.int64!index", [1048576]]],
            0.0,
        ],
        "kwargs": {},
        "key": [1048576],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch._unsafe_masked_index``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch._unsafe_masked_index")
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
