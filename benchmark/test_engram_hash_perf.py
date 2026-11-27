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

import flag_gems
from tests.test_engram_hash import make_inputs

from . import base


def inputs(shape, dtype, device):
    args, kwargs = make_inputs(shape, device)
    yield (*args, kwargs)


class AtlasBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


@pytest.mark.engram_hash
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="requires Atlas A5",
)
def test_engram_hash_perf():
    reference = pytest.importorskip(
        "tile_kernels.engram.engram_hash_kernel"
    ).engram_hash
    bench = AtlasBenchmark(
        op_name="engram_hash", input_fn=inputs, torch_op=reference, dtypes=[torch.int32]
    )
    bench.set_gems(flag_gems.engram_hash)
    bench.run()
