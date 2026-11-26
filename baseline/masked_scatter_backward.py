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

"""NV baseline for ``masked_scatter_backward``.

Native: ``aten::masked_scatter_backward``

Grid: 12 collection point(s) over 3 dtype(s), taken from the shapes and inputs
this operator's benchmark uses (M, N).
"""

import _common
import torch

OP_NAME = "masked_scatter_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
ATEN_NAME = "masked_scatter_backward"

NATIVE_UNAVAILABLE_REASON = None

# One entry per collection point: the arguments of a single benchmark case.
# "T" is a tensor of the case dtype, "T:<dtype>" one pinned to another dtype,
# and "T!<kind>" one whose values are constrained (see _common.make_tensor).
# Taken from this operator's own FlagGems benchmark, so a collected record
# matches the case `--base-data` compares it against.
GRID = [
    {
        "args": [["T", [64, 64]], ["T:torch.bool", [64, 64]], [64, 64]],
        "kwargs": {},
        "key": [64, 64],
    },
    {
        "args": [["T", [256, 256]], ["T:torch.bool", [256, 256]], [256, 256]],
        "kwargs": {},
        "key": [256, 256],
    },
    {
        "args": [["T", [1024, 1024]], ["T:torch.bool", [1024, 1024]], [1024, 1024]],
        "kwargs": {},
        "key": [1024, 1024],
    },
    {
        "args": [["T", [4096, 4096]], ["T:torch.bool", [4096, 4096]], [4096, 4096]],
        "kwargs": {},
        "key": [4096, 4096],
    },
    {
        "args": [["T", [1024, 65536]], ["T:torch.bool", [1024, 65536]], [1024, 65536]],
        "kwargs": {},
        "key": [1024, 65536],
    },
    {
        "args": [["T", [10000, 256]], ["T:torch.bool", [10000, 256]], [10000, 256]],
        "kwargs": {},
        "key": [10000, 256],
    },
    {
        "args": [
            ["T", [10000, 65536]],
            ["T:torch.bool", [10000, 65536]],
            [10000, 65536],
        ],
        "kwargs": {},
        "key": [10000, 65536],
    },
    {
        "args": [["T", [17, 17]], ["T:torch.bool", [17, 17]], [17, 17]],
        "kwargs": {},
        "key": [17, 17],
    },
    {
        "args": [["T", [32, 128]], ["T:torch.bool", [32, 128]], [32, 128]],
        "kwargs": {},
        "key": [32, 128],
    },
    {
        "args": [["T", [128, 64]], ["T:torch.bool", [128, 64]], [128, 64]],
        "kwargs": {},
        "key": [128, 64],
    },
    {
        "args": [["T", [256, 64]], ["T:torch.bool", [256, 64]], [256, 64]],
        "kwargs": {},
        "key": [256, 64],
    },
    {
        "args": [["T", [1024, 2048]], ["T:torch.bool", [1024, 2048]], [1024, 2048]],
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
