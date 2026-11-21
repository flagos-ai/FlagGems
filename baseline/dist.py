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

"""NVIDIA baseline for dist (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.dist``; the composed fallback that FlagGems uses on Ascend for
``p = +/-inf`` and non-standard ``p`` is not needed here, since baselines are
collected on NVIDIA and torch handles every benchmarked ``p`` on CUDA. Shapes
(the fixed 1-D ``SHAPES`` list), dtypes and the ``p`` list (``-inf, inf, 0, 1,
2, 6.6``) are copied from ``benchmark/test_dist.py``.
"""

import torch

OP_NAME = "dist"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False


# benchmark/test_dist.py::DistBenchmark.SHAPES
_CASES = [
    (16,),
    (64,),
    (128,),
    (512,),
    (1024,),  # small
    (16384,),  # single-launch path upper bound
    (2**20,),  # medium
    (2**24,),
    (2**28,),  # large
]

# benchmark/test_dist.py::_input_fn p values
_P_LIST = ("-inf", "inf", "0.0", "1.0", "2.0", "6.6")


def native():
    return torch.dist


def grid():
    return [{"shape": shape, "p": p} for shape in _CASES for p in _P_LIST]


def build_inputs(binding, dtype, device):
    inp1 = torch.randn(binding["shape"], dtype=dtype, device=device)
    inp2 = torch.randn(binding["shape"], dtype=dtype, device=device)
    return (inp1, inp2), {"p": float(binding["p"])}


def key_shape(binding):
    shape = "x".join(str(dim) for dim in binding["shape"])
    return f"{shape}_p{binding['p']}"


def config(binding, dtype):
    # dist reduces every element to a 0-dim scalar; both operands share the case
    # shape and dtype.
    shape = list(binding["shape"])
    return {
        "inputs": {
            "inp1": {"shape": shape, "dtype": str(dtype)},
            "inp2": {"shape": shape, "dtype": str(dtype)},
        },
        "outputs": {"out": {"shape": [], "dtype": str(dtype)}},
        "dims": {"p": binding["p"]},
    }
