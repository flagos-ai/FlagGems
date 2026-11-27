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

"""NV baseline for ``nonzero_static``.

Native: ``torch.nonzero_static``

Grid: 9 collection point(s) over 2 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (shape, nnz_ratio, size, fill_value).
"""

import _common
import torch

OP_NAME = "nonzero_static"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "nonzero_static"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [["T", [1024]]],
        "kwargs": {"size": 128, "fill_value": -1},
        "key": [1024],
    },
    {
        "args": [["T", [16384]]],
        "kwargs": {"size": 128, "fill_value": -1},
        "key": [16384],
    },
    {
        "args": [["T", [262144]]],
        "kwargs": {"size": 1024, "fill_value": -1},
        "key": [262144],
    },
    {
        "args": [["T", [1048576]]],
        "kwargs": {"size": 1024, "fill_value": -1},
        "key": [1048576],
    },
    {
        "args": [["T", [32, 1024]]],
        "kwargs": {"size": 1024, "fill_value": -1},
        "key": [32, 1024],
    },
    {
        "args": [["T", [128, 4096]]],
        "kwargs": {"size": 4096, "fill_value": -1},
        "key": [128, 4096],
    },
    {
        "args": [["T", [1024, 4096]]],
        "kwargs": {"size": 1024, "fill_value": -1},
        "key": [1024, 4096],
    },
    {
        "args": [["T", [16, 64, 64]]],
        "kwargs": {"size": 1024, "fill_value": -1},
        "key": [16, 64, 64],
    },
    {
        "args": [["T", [32, 128, 128]]],
        "kwargs": {"size": 4096, "fill_value": -1},
        "key": [32, 128, 128],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch.nonzero_static``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch.nonzero_static")
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
