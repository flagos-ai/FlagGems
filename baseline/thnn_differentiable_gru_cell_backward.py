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

"""NV baseline for ``thnn_differentiable_gru_cell_backward``.

Native: ``aten::_thnn_differentiable_gru_cell_backward``

Grid: 9 collection point(s) over 4 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import _common
import torch

OP_NAME = "thnn_differentiable_gru_cell_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.float64]
IS_INPLACE = False
ATEN_NAME = "_thnn_differentiable_gru_cell_backward"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [
            ["T", [1, 32]],
            ["T", [1, 96]],
            ["T", [1, 96]],
            ["T", [1, 32]],
            None,
            None,
        ],
        "kwargs": {},
        "key": [1, 32],
    },
    {
        "args": [
            ["T", [8, 64]],
            ["T", [8, 192]],
            ["T", [8, 192]],
            ["T", [8, 64]],
            ["T", [192]],
            ["T", [192]],
        ],
        "kwargs": {},
        "key": [8, 64],
    },
    {
        "args": [
            ["T", [32, 128]],
            ["T", [32, 384]],
            ["T", [32, 384]],
            ["T", [32, 128]],
            None,
            None,
        ],
        "kwargs": {},
        "key": [32, 128],
    },
    {
        "args": [
            ["T", [64, 256]],
            ["T", [64, 768]],
            ["T", [64, 768]],
            ["T", [64, 256]],
            ["T", [768]],
            ["T", [768]],
        ],
        "kwargs": {},
        "key": [64, 256],
    },
    {
        "args": [
            ["T", [128, 512]],
            ["T", [128, 1536]],
            ["T", [128, 1536]],
            ["T", [128, 512]],
            None,
            None,
        ],
        "kwargs": {},
        "key": [128, 512],
    },
    {
        "args": [
            ["T", [256, 1024]],
            ["T", [256, 3072]],
            ["T", [256, 3072]],
            ["T", [256, 1024]],
            ["T", [3072]],
            ["T", [3072]],
        ],
        "kwargs": {},
        "key": [256, 1024],
    },
    {
        "args": [
            ["T", [512, 1024]],
            ["T", [512, 3072]],
            ["T", [512, 3072]],
            ["T", [512, 1024]],
            None,
            None,
        ],
        "kwargs": {},
        "key": [512, 1024],
    },
    {
        "args": [
            ["T", [512, 2048]],
            ["T", [512, 6144]],
            ["T", [512, 6144]],
            ["T", [512, 2048]],
            ["T", [6144]],
            ["T", [6144]],
        ],
        "kwargs": {},
        "key": [512, 2048],
    },
    {
        "args": [
            ["T", [1024, 2048]],
            ["T", [1024, 6144]],
            ["T", [1024, 6144]],
            ["T", [1024, 2048]],
            ["T", [6144]],
            ["T", [6144]],
        ],
        "kwargs": {},
        "key": [1024, 2048],
    },
]


def native():
    """Resolve the NV native callable, or None when this build lacks it."""
    global NATIVE_UNAVAILABLE_REASON
    op, reason = _common.resolve(ATEN_NAME)
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
