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

"""NV baseline for ``linalg_lstsq``.

Native: ``torch._C._linalg.linalg_lstsq``

Grid: 23 collection point(s) over 2 dtype(s), taken from the shapes and inputs
this operator's benchmark uses ((*B), M, N  (M>=N tall, or M<N wide)).
"""

import torch

import _common

OP_NAME = "linalg_lstsq"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
ATEN_NAME = "linalg_lstsq"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [['T', [256, 32]], ['T', [256]]],
        "kwargs": {'driver': 'gels'},
        "key": [256, 32],
    },
    {
        "args": [['T', [1024, 16]], ['T', [1024]]],
        "kwargs": {'driver': 'gels'},
        "key": [1024, 16],
    },
    {
        "args": [['T', [4096, 8]], ['T', [4096]]],
        "kwargs": {'driver': 'gels'},
        "key": [4096, 8],
    },
    {
        "args": [['T', [8, 4096, 8]], ['T', [8, 4096]]],
        "kwargs": {'driver': 'gels'},
        "key": [8, 4096, 8],
    },
    {
        "args": [['T', [64, 2048, 16]], ['T', [64, 2048]]],
        "kwargs": {'driver': 'gels'},
        "key": [64, 2048, 16],
    },
    {
        "args": [['T', [16, 8192, 16]], ['T', [16, 8192]]],
        "kwargs": {'driver': 'gels'},
        "key": [16, 8192, 16],
    },
    {
        "args": [['T', [8, 16, 512]], ['T', [8, 16]]],
        "kwargs": {'driver': 'gels'},
        "key": [8, 16, 512],
    },
    {
        "args": [['T', [8, 64, 256]], ['T', [8, 64]]],
        "kwargs": {'driver': 'gels'},
        "key": [8, 64, 256],
    },
    {
        "args": [['T', [16, 16, 1024]], ['T', [16, 16]]],
        "kwargs": {'driver': 'gels'},
        "key": [16, 16, 1024],
    },
    {
        "args": [['T', [64, 32, 256]], ['T', [64, 32]]],
        "kwargs": {'driver': 'gels'},
        "key": [64, 32, 256],
    },
    {
        "args": [['T', [64, 1024]], ['T', [64]]],
        "kwargs": {'driver': 'gels'},
        "key": [64, 1024],
    },
    {
        "args": [['T', [128, 512]], ['T', [128]]],
        "kwargs": {'driver': 'gels'},
        "key": [128, 512],
    },
    {
        "args": [['T', [8, 64, 1024]], ['T', [8, 64]]],
        "kwargs": {'driver': 'gels'},
        "key": [8, 64, 1024],
    },
    {
        "args": [['T', [16, 32, 2048]], ['T', [16, 32]]],
        "kwargs": {'driver': 'gels'},
        "key": [16, 32, 2048],
    },
    {
        "args": [['T', [256, 256]], ['T', [256]]],
        "kwargs": {'driver': 'gels'},
        "key": [256, 256],
    },
    {
        "args": [['T', [512, 512]], ['T', [512]]],
        "kwargs": {'driver': 'gels'},
        "key": [512, 512],
    },
    {
        "args": [['T', [1024, 1024]], ['T', [1024]]],
        "kwargs": {'driver': 'gels'},
        "key": [1024, 1024],
    },
    {
        "args": [['T', [2048, 2048]], ['T', [2048]]],
        "kwargs": {'driver': 'gels'},
        "key": [2048, 2048],
    },
    {
        "args": [['T', [8, 256, 256]], ['T', [8, 256]]],
        "kwargs": {'driver': 'gels'},
        "key": [8, 256, 256],
    },
    {
        "args": [['T', [8, 512, 512]], ['T', [8, 512]]],
        "kwargs": {'driver': 'gels'},
        "key": [8, 512, 512],
    },
    {
        "args": [['T', [4096, 512]], ['T', [4096]]],
        "kwargs": {'driver': 'gels'},
        "key": [4096, 512],
    },
    {
        "args": [['T', [512, 1024]], ['T', [512]]],
        "kwargs": {'driver': 'gels'},
        "key": [512, 1024],
    },
    {
        "args": [['T', [1024, 2048]], ['T', [1024]]],
        "kwargs": {'driver': 'gels'},
        "key": [1024, 2048],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it.

    The benchmark times ``torch._C._linalg.linalg_lstsq``, whose signature differs from the raw ATen
    overload (it wraps extra flags or a packed return). The grid holds that
    call's arguments, so the baseline has to time the same entry point.
    """
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve_path("torch._C._linalg.linalg_lstsq")
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
