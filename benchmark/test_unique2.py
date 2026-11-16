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

import pytest
import torch

from . import base, consts

# aten::_unique2 sorts the flattened input, so keep shapes well under sort's
# 2^30 element limit.
_UNIQUE2_SHAPES = [
    (1024,),
    (4096,),
    (16384,),
    (65536,),
    (262144,),
]

# The four return-flag combinations are benchmarked as cases of the single
# unique2 operator identity rather than as separate pseudo operators.
_FLAG_COMBOS = [
    {"sorted": True, "return_inverse": False, "return_counts": False},
    {"sorted": True, "return_inverse": True, "return_counts": False},
    {"sorted": True, "return_inverse": False, "return_counts": True},
    {"sorted": True, "return_inverse": True, "return_counts": True},
]


class Unique2Benchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _UNIQUE2_SHAPES

    def set_more_shapes(self):
        # Shapes are fully specified by DEFAULT_SHAPES.
        return []


def unique2_input_fn(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
        # Quantize to create duplicates.
        inp = torch.round(inp * 10) / 10
    else:
        # Small range ensures duplicates for integer inputs.
        high = min(1000, shape[0] // 10) if shape[0] > 100 else 10
        inp = torch.randint(0, high, shape, dtype=dtype, device=device)
    # Emit each return-flag combination as its own benchmark case.
    for flags in _FLAG_COMBOS:
        yield inp, dict(flags)


@pytest.mark.underscore_unique2
def test_perf_unique2():
    bench = Unique2Benchmark(
        input_fn=unique2_input_fn,
        op_name="unique2",
        torch_op=torch._unique2,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()
