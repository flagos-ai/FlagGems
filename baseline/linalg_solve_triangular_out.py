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

"""NV baseline for ``linalg_solve_triangular_out``.

Native: ``torch._C._linalg.linalg_solve_triangular``

Grid: 7 collection point(s) over 2 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import torch

import _common

OP_NAME = "linalg_solve_triangular_out"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
ATEN_NAME = "linalg_solve_triangular.out"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [8, 8]], ['T', [8, 16]]],
        "kwargs": {'upper': False, 'out': ['T', [8, 16]]},
        "key": [8, 8],
    },
    {
        "args": [['T', [16, 16]], ['T', [16, 32]]],
        "kwargs": {'upper': False, 'out': ['T', [16, 32]]},
        "key": [16, 16],
    },
    {
        "args": [['T', [32, 32]], ['T', [32, 64]]],
        "kwargs": {'upper': False, 'out': ['T', [32, 64]]},
        "key": [32, 32],
    },
    {
        "args": [['T', [64, 64]], ['T', [64, 128]]],
        "kwargs": {'upper': False, 'out': ['T', [64, 128]]},
        "key": [64, 64],
    },
    {
        "args": [['T', [128, 128]], ['T', [128, 256]]],
        "kwargs": {'upper': False, 'out': ['T', [128, 256]]},
        "key": [128, 128],
    },
    {
        "args": [['T', [256, 256]], ['T', [256, 512]]],
        "kwargs": {'upper': False, 'out': ['T', [256, 512]]},
        "key": [256, 256],
    },
    {
        "args": [['T', [512, 512]], ['T', [512, 256]]],
        "kwargs": {'upper': False, 'out': ['T', [512, 256]]},
        "key": [512, 512],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch._C._linalg.linalg_solve_triangular``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch._C._linalg.linalg_solve_triangular")
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
