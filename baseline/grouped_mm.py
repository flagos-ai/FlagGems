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

"""NV baseline for ``grouped_mm``.

Native: ``torch._grouped_mm``

Grid: 3 collection point(s) over 1 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (Groups, N, K).
"""

import _common
import torch

OP_NAME = "grouped_mm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "_grouped_mm"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [
            ["T", [102647, 2048]],
            ["T", [16, 2048, 512]],
            ["T:torch.int32!index", [16]],
        ],
        "kwargs": {},
        "key": [102647, 2048],
    },
    {
        "args": [
            ["T", [125941, 2048]],
            ["T", [16, 2048, 2560]],
            ["T:torch.int32!index", [16]],
        ],
        "kwargs": {},
        "key": [125941, 2048],
    },
    {
        "args": [
            ["T", [538225, 128]],
            ["T", [64, 128, 2048]],
            ["T:torch.int32!index", [64]],
        ],
        "kwargs": {},
        "key": [538225, 128],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch._grouped_mm``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch._grouped_mm")
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
