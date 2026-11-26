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

"""NV baseline for ``upsample_nearest1d``.

Native: ``torch._C._nn.upsample_nearest1d``

Grid: 5 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (N, C, H, W).
"""

import _common
import torch

OP_NAME = "upsample_nearest1d"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "upsample_nearest1d"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [],
        "kwargs": {
            "input": ["T", [1, 3, 262144]],
            "output_size": [524288],
            "scales": None,
        },
        "key": [1, 3, 262144],
    },
    {
        "args": [],
        "kwargs": {
            "input": ["T", [8, 16, 16384]],
            "output_size": [32768],
            "scales": None,
        },
        "key": [8, 16, 16384],
    },
    {
        "args": [],
        "kwargs": {
            "input": ["T", [2, 3, 1048576]],
            "output_size": [2097152],
            "scales": None,
        },
        "key": [2, 3, 1048576],
    },
    {
        "args": [],
        "kwargs": {
            "input": ["T", [16, 16, 262144]],
            "output_size": [524288],
            "scales": None,
        },
        "key": [16, 16, 262144],
    },
    {
        "args": [],
        "kwargs": {
            "input": ["T", [16, 16, 1048576]],
            "output_size": [2097152],
            "scales": None,
        },
        "key": [16, 16, 1048576],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch._C._nn.upsample_nearest1d``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch._C._nn.upsample_nearest1d")
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
