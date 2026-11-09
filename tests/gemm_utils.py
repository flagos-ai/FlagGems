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

if __package__ == "flaggems_tests":
    from flaggems_benchmark.gemm_shapes import accuracy_cases, accuracy_shapes
else:
    from benchmark.gemm_shapes import accuracy_cases, accuracy_shapes

GEMM_DTYPES = [
    torch.float16,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not flag_gems.runtime.device.support_bf16,
            reason="this backend does not support bfloat16",
        ),
    ),
]

__all__ = ["GEMM_DTYPES", "accuracy_cases", "accuracy_shapes"]
