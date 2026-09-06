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

from . import base


@pytest.mark.atanh_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_atanh_():
    # bf16 is excluded: vendor native a.atanh_() (xdnn_pytorch_wrapper
    # atanh.cpp:32) reports [NOT IMPLEMENTED] for kbfloat16 on XPU, so the
    # benchmark's latency_base (native reference) has no bf16 baseline.
    # bf16 correctness is covered by tests/test_atanh_.py --ref cpu.
    # Same pattern as benchmark/test_sinh.py / test_mish.py / test_cosh.py.
    bench = base.UnaryPointwiseBenchmark(
        op_name="atanh_",
        torch_op=lambda a: a.atanh_(),
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()
