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

from . import base, consts


def npu_log_normal_(self, mean=1.0, std=2.0):
    # Ascend NPU does not support log_normal_ natively, emulate it with supported torch_npu ops.
    self.normal_(mean=0.0, std=1.0)
    self.mul_(std)
    self.add_(mean)
    self.exp_()
    return self


@pytest.mark.log_normal_
def test_log_normal_():
    torch_op = (
        npu_log_normal_ if flag_gems.device == "npu" else torch.Tensor.log_normal_
    )

    bench = base.GenericBenchmark(
        op_name="log_normal_",
        torch_op=torch_op,
        gems_op=flag_gems.log_normal_,
        input_fn=base.unary_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
