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

"""NV baseline for ``adaptive_max_pool3d``.

Native: ``torch.nn.functional.adaptive_max_pool3d``

Grid: 13 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses ([input_shape, output_size]).
"""

import torch

import _common

OP_NAME = "adaptive_max_pool3d"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "adaptive_max_pool3d"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [2, 256, 2, 14, 14]], [2, 14, 14]],
        "kwargs": {'return_indices': True},
        "key": [2, 256, 2, 14, 14],
    },
    {
        "args": [['T', [1, 256, 1, 64, 64]], [1, 1, 1]],
        "kwargs": {'return_indices': False},
        "key": [1, 256, 1, 64, 64],
    },
    {
        "args": [['T', [1, 4, 8, 24, 32]], [1, 1, 1]],
        "kwargs": {'return_indices': False},
        "key": [1, 4, 8, 24, 32],
    },
    {
        "args": [['T', [1, 8, 64, 256, 256]], [1, 7, 7]],
        "kwargs": {'return_indices': False},
        "key": [1, 8, 64, 256, 256],
    },
    {
        "args": [['T', [8, 64, 16, 112, 112]], [1, 7, 7]],
        "kwargs": {'return_indices': False},
        "key": [8, 64, 16, 112, 112],
    },
    {
        "args": [['T', [1, 256, 64, 112, 112]], [1, 112, 112]],
        "kwargs": {'return_indices': False},
        "key": [1, 256, 64, 112, 112],
    },
    {
        "args": [['T', [4, 128, 32, 64, 64]], [8, 64, 64]],
        "kwargs": {'return_indices': False},
        "key": [4, 128, 32, 64, 64],
    },
    {
        "args": [['T', [2, 128, 4, 28, 28]], [4, 14, 14]],
        "kwargs": {'return_indices': False},
        "key": [2, 128, 4, 28, 28],
    },
    {
        "args": [['T', [2, 640, 64, 42, 72]], [16, 14, 14]],
        "kwargs": {'return_indices': False},
        "key": [2, 640, 64, 42, 72],
    },
    {
        "args": [['T', [32, 512, 8, 8, 8]], [4, 4, 4]],
        "kwargs": {'return_indices': False},
        "key": [32, 512, 8, 8, 8],
    },
    {
        "args": [['T', [1, 768, 160, 32, 32]], [8, 32, 32]],
        "kwargs": {'return_indices': False},
        "key": [1, 768, 160, 32, 32],
    },
    {
        "args": [['T', [1, 1280, 48, 36, 50]], [8, 8, 8]],
        "kwargs": {'return_indices': False},
        "key": [1, 1280, 48, 36, 50],
    },
    {
        "args": [['T', [1, 3, 16, 224, 224]], [8, 7, 7]],
        "kwargs": {'return_indices': False},
        "key": [1, 3, 16, 224, 224],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch.nn.functional.adaptive_max_pool3d``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch.nn.functional.adaptive_max_pool3d")
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
