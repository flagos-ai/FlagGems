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

"""NVIDIA baseline for log_normal (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.ops.aten.log_normal``; the composed ``normal_`` / ``mul_`` / ``add_`` /
``exp_`` reference that ``benchmark/test_log_normal.py`` substitutes on NPU is
not needed here, since baselines are collected on NVIDIA. Shapes
(``benchmark/core_shapes.yaml::Benchmark``),
dtypes and the default ``mean=1.0`` / ``std=2.0`` are copied from
``benchmark/test_log_normal.py``.
"""

import torch

OP_NAME = "log_normal"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False


# benchmark/core_shapes.yaml::Benchmark
_CASES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]


def native():
    return torch.ops.aten.log_normal


def grid():
    return [{"shape": case} for case in _CASES]


def build_inputs(binding, dtype, device):
    x = torch.randn(binding["shape"], dtype=dtype, device=device)
    return (x,), {}


def key_shape(binding):
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    # log_normal keeps the case shape and dtype; mean/std are the fixed defaults.
    shape = list(binding["shape"])
    return {
        "inputs": {
            "x": {"shape": shape, "dtype": str(dtype)},
        },
        "outputs": {"out": {"shape": shape, "dtype": str(dtype)}},
        "dims": {"mean": 1.0, "std": 2.0},
    }
