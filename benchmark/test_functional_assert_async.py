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

from . import consts


@pytest.mark.functional_assert_async
def test_functional_assert_async():
    """Benchmark FlagGems implementation only (PyTorch has no CUDA reference)"""

    for dtype in consts.INT_DTYPES + consts.FLOAT_DTYPES:
        inp = torch.ones(1, dtype=dtype, device=flag_gems.device)
        dep_token = torch.empty(0, dtype=dtype, device=flag_gems.device)

        # Warmup
        with flag_gems.use_gems():
            for _ in range(10):
                _ = torch.ops.aten._functional_assert_async.msg(inp, "test", dep_token)

        # Simple timing (no comparison since PyTorch has no CUDA impl)
        import time

        torch.cuda.synchronize()
        start = time.perf_counter()
        with flag_gems.use_gems():
            for _ in range(100):
                _ = torch.ops.aten._functional_assert_async.msg(inp, "test", dep_token)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(
            f"  {dtype}: {elapsed*1000:.3f}ms for 100 iters, {elapsed*10:.3f}us per call"
        )

    print("  Benchmark completed (no PyTorch CUDA reference available for comparison)")
