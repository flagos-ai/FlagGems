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

import random

import pytest
import torch

import flag_gems

from . import base, consts


def replication_pad3d_input_fn(shape, dtype, device):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    p = random.randint(1, 3)
    padding = (p, p, p, p, p, p)

    yield input_tensor, {"padding": padding}


class ReplicationPad3dBenchmark(base.GenericBenchmarkExcluse3D):
    def set_shapes(self, shape_file_path=None):
        replication_pad3d_shapes = [
            (1, 3, 16, 256, 256),
            (4, 16, 32, 64, 64),
            (8, 64, 8, 32, 32),
            (2, 32, 16, 128, 128),
            (1, 1, 64, 128, 128),
        ]
        self.shapes = replication_pad3d_shapes

    def set_more_shapes(self):
        return []


@pytest.mark.replication_pad3d
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_replication_pad3d():
    def torch_replication_pad3d(input, padding):
        return torch.nn.functional.pad(input, padding, mode="replicate")

    def gems_wrapper(input, padding):
        return flag_gems.replication_pad3d(input, padding)

    bench = ReplicationPad3dBenchmark(
        input_fn=replication_pad3d_input_fn,
        op_name="replication_pad3d",
        torch_op=torch_replication_pad3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_wrapper)
    bench.run()
