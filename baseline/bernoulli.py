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

"""NV baseline for ``bernoulli``.

Native: ``torch.bernoulli``

Grid: 12 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import _common
import torch

OP_NAME = "bernoulli"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "bernoulli"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [["T!unit_interval", [1073741824]]],
        "kwargs": {},
        "key": [1073741824],
    },
    {
        "args": [["T!unit_interval", [64, 64]]],
        "kwargs": {},
        "key": [64, 64],
    },
    {
        "args": [["T!unit_interval", [4096, 4096]]],
        "kwargs": {},
        "key": [4096, 4096],
    },
    {
        "args": [["T!unit_interval", [64, 512, 512]]],
        "kwargs": {},
        "key": [64, 512, 512],
    },
    {
        "args": [["T!unit_interval", [1024, 1024, 1024]]],
        "kwargs": {},
        "key": [1024, 1024, 1024],
    },
    {
        "args": [["T!unit_interval", [268435456]]],
        "kwargs": {},
        "key": [268435456],
    },
    {
        "args": [["T!unit_interval", [10000, 1]]],
        "kwargs": {},
        "key": [10000, 1],
    },
    {
        "args": [["T!unit_interval", [10000, 256]]],
        "kwargs": {},
        "key": [10000, 256],
    },
    {
        "args": [["T!unit_interval", [10000, 65536]]],
        "kwargs": {},
        "key": [10000, 65536],
    },
    {
        "args": [["T!unit_interval", [100, 1, 100]]],
        "kwargs": {},
        "key": [100, 1, 100],
    },
    {
        "args": [["T!unit_interval", [100, 256, 100]]],
        "kwargs": {},
        "key": [100, 256, 100],
    },
    {
        "args": [["T!unit_interval", [100, 65536, 100]]],
        "kwargs": {},
        "key": [100, 65536, 100],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch.bernoulli``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch.bernoulli")
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
