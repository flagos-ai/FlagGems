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

from . import base


@pytest.mark.sinh
def test_sinh():
    # bf16 is excluded: vendor native torch.sinh (xdnn_pytorch_wrapper
    # sinh.cpp:30) reports [NOT IMPLEMENTED] for kbfloat16 on XPU, so the
    # benchmark's latency_base (native reference) has no bf16 baseline.
    # bf16 correctness is covered by tests/test_sinh.py --ref cpu.
    # Same pattern as benchmark/test_mish.py / test_cosh.py.
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinh",
        torch_op=torch.sinh,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


@pytest.mark.sinh_
def test_sinh_inplace():
    # bf16 is excluded: vendor native torch.sinh_ (xdnn_pytorch_wrapper
    # sinh.cpp:30) reports [NOT IMPLEMENTED] for kbfloat16 on XPU, so the
    # benchmark's latency_base (native reference) has no bf16 baseline.
    # bf16 correctness is covered by tests/test_sinh.py --ref cpu.
    # Same pattern as benchmark/test_mish.py / test_cosh.py.
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinh_",
        torch_op=lambda a: a.sinh_(),
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()
