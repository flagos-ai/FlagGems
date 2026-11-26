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

"""NV baseline for ``linalg_lu_factor_ex_out``.

Native: ``torch._C._linalg.linalg_lu_factor_ex``

Grid: 13 collection point(s) over 2 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (input shape, pivot, check_errors).
"""

import torch

import _common

OP_NAME = "linalg_lu_factor_ex_out"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
ATEN_NAME = "linalg_lu_factor_ex.out"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [16, 16]]],
        "kwargs": {'out': [['T', [16, 16]], ['T:torch.int32!index', [16]], ['T:torch.int32!index', []]]},
        "key": [16, 16],
    },
    {
        "args": [['T', [32, 32]]],
        "kwargs": {'out': [['T', [32, 32]], ['T:torch.int32!index', [32]], ['T:torch.int32!index', []]]},
        "key": [32, 32],
    },
    {
        "args": [['T', [64, 64]]],
        "kwargs": {'out': [['T', [64, 64]], ['T:torch.int32!index', [64]], ['T:torch.int32!index', []]]},
        "key": [64, 64],
    },
    {
        "args": [['T', [128, 128]]],
        "kwargs": {'out': [['T', [128, 128]], ['T:torch.int32!index', [128]], ['T:torch.int32!index', []]]},
        "key": [128, 128],
    },
    {
        "args": [['T', [256, 256]]],
        "kwargs": {'out': [['T', [256, 256]], ['T:torch.int32!index', [256]], ['T:torch.int32!index', []]]},
        "key": [256, 256],
    },
    {
        "args": [['T', [1024, 512]]],
        "kwargs": {'out': [['T', [1024, 512]], ['T:torch.int32!index', [512]], ['T:torch.int32!index', []]]},
        "key": [1024, 512],
    },
    {
        "args": [['T', [32, 16]]],
        "kwargs": {'out': [['T', [32, 16]], ['T:torch.int32!index', [16]], ['T:torch.int32!index', []]]},
        "key": [32, 16],
    },
    {
        "args": [['T', [16, 32]]],
        "kwargs": {'out': [['T', [16, 32]], ['T:torch.int32!index', [16]], ['T:torch.int32!index', []]]},
        "key": [16, 32],
    },
    {
        "args": [['T', [128, 64]]],
        "kwargs": {'out': [['T', [128, 64]], ['T:torch.int32!index', [64]], ['T:torch.int32!index', []]]},
        "key": [128, 64],
    },
    {
        "args": [['T', [64, 128]]],
        "kwargs": {'out': [['T', [64, 128]], ['T:torch.int32!index', [64]], ['T:torch.int32!index', []]]},
        "key": [64, 128],
    },
    {
        "args": [['T', [4, 32, 32]]],
        "kwargs": {'out': [['T', [4, 32, 32]], ['T:torch.int32!index', [4, 32]], ['T:torch.int32!index', [4]]]},
        "key": [4, 32, 32],
    },
    {
        "args": [['T', [128, 16, 16]]],
        "kwargs": {'out': [['T', [128, 16, 16]], ['T:torch.int32!index', [128, 16]], ['T:torch.int32!index', [128]]]},
        "key": [128, 16, 16],
    },
    {
        "args": [['T', [1024, 512, 512]]],
        "kwargs": {
            'out': [['T', [1024, 512, 512]], ['T:torch.int32!index', [1024, 512]], ['T:torch.int32!index', [1024]]],
        },
        "key": [1024, 512, 512],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch._C._linalg.linalg_lu_factor_ex``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch._C._linalg.linalg_lu_factor_ex")
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
