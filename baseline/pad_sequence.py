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

"""NV baseline for ``pad_sequence``.

Native: ``torch.nn.utils.rnn.pad_sequence``

Grid: 7 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import torch

import _common

OP_NAME = "pad_sequence"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "pad_sequence"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [[['T', [64, 8]], ['T', [32, 8]]]],
        "kwargs": {'batch_first': False, 'padding_value': 0.0},
        "key": [64, 8],
    },
    {
        "args": [[['T', [128, 16]], ['T', [256, 16]], ['T', [100, 16]], ['T', [200, 16]]]],
        "kwargs": {'batch_first': False, 'padding_value': 0.0},
        "key": [128, 16],
    },
    {
        "args": [[['T', [512, 768]], ['T', [256, 768]], ['T', [400, 768]], ['T', [128, 768]]]],
        "kwargs": {'batch_first': False, 'padding_value': 0.0},
        "key": [512, 768],
    },
    {
        "args": [[['T', [1024, 1024]], ['T', [768, 1024]], ['T', [512, 1024]], ['T', [256, 1024]]]],
        "kwargs": {'batch_first': False, 'padding_value': 0.0},
        "key": [1024, 1024],
    },
    {
        "args": [[['T', [2048, 4096]], ['T', [1024, 4096]], ['T', [1536, 4096]], ['T', [512, 4096]]]],
        "kwargs": {'batch_first': False, 'padding_value': 0.0},
        "key": [2048, 4096],
    },
    {
        "args": [
            [
                ['T', [512, 512]],
                ['T', [384, 512]],
                ['T', [256, 512]],
                ['T', [128, 512]],
                ['T', [448, 512]],
                ['T', [320, 512]],
                ['T', [192, 512]],
                ['T', [64, 512]],
                ['T', [400, 512]],
                ['T', [300, 512]],
                ['T', [200, 512]],
                ['T', [100, 512]],
                ['T', [256, 512]],
                ['T', [128, 512]],
                ['T', [96, 512]],
                ['T', [32, 512]],
            ],
        ],
        "kwargs": {'batch_first': False, 'padding_value': 0.0},
        "key": [512, 512],
    },
    {
        "args": [
            [
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
                ['T', [512, 1024]],
            ],
        ],
        "kwargs": {'batch_first': False, 'padding_value': 0.0},
        "key": [512, 1024],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch.nn.utils.rnn.pad_sequence``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch.nn.utils.rnn.pad_sequence")
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
