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

"""Scope the measured kernels without changing default precision policy."""
import torch


def use_tuned_gemm(a, b):
    if a.device != b.device or a.dtype != b.dtype:
        return False
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if a.dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32:
        return False
    for tensor in (a, b):
        if any(s < 0 for s in tensor.stride()):
            return False
        span = sum(max(n - 1, 0) * s for n, s in zip(tensor.shape, tensor.stride()))
        if span >= 2**31:
            return False
    return True
