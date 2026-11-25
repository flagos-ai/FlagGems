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

"""NVIDIA baseline for polygamma (n=1, torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.polygamma`` called with ``n=1`` (trigamma), matching the benchmark's
``_torch_polygamma1``. The composed trigamma fallback used by FlagGems on
non-NVIDIA vendors is not needed here, since baselines are collected on NVIDIA.
Shapes and dtypes are copied from ``benchmark/test_polygamma.py``.
"""

import torch

OP_NAME = "polygamma"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/core_shapes.yaml::UnaryPointwiseBenchmark
_CASES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]


def _polygamma1(x):
    return torch.polygamma(1, x)


def native():
    return _polygamma1


def grid():
    return [{"shape": case} for case in _CASES]


def build_inputs(binding, dtype, device):
    x = torch.randn(binding["shape"], dtype=dtype, device=device)
    return (x,), {}


def key_shape(binding):
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    return {
        "inputs": {"input": {"shape": shape, "dtype": str(dtype)}},
        "outputs": {"out": {"shape": shape, "dtype": str(dtype)}},
        "dims": {"n": 1},
    }
