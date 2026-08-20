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

import torch
from torch.library import opcheck

import flag_gems  # noqa: F401

inputs = [
    (
        torch.randn(2, 8, 4, device="cuda:0"),
        torch.randn(4, device="cuda:0"),
    ),
    (
        torch.randn(3, 8, 4, device="cuda:0", requires_grad=True),
        torch.randn(3, 1, 4, device="cuda:0"),
    ),
    (
        torch.randn(2, 8, 4, device="cuda:0"),
        torch.randn(1, 4, device="cuda:0", requires_grad=True),
    ),
    (
        torch.randn(2, 1, 4, device="cuda:0", requires_grad=True),
        torch.randn(2, 8, 1, device="cuda:0", requires_grad=True),
    ),
]

for arg in inputs:
    opcheck(
        torch.ops.flag_gems.add_tensor.default,
        arg,
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_static",
            "test_aot_dispatch_dynamic",
        ),
    )
