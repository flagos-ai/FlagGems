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

import os

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.randn
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_randn(shape, dtype):
    if flag_gems.vendor_name in ["cambricon", "iluvatar"]:
        torch.manual_seed(42)

    res_out = flag_gems.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = utils.to_reference(res_out).float()
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)

    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.randn
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("nvidia", "thead"),
    reason="Checks the generic CUDA/PPU randn implementation and CUDA RNG state.",
)
@pytest.mark.parametrize("numel", [0, 1, 511, 512, 513, 2047, 2048, 2049])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_randn_seeded_tail(numel, dtype):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1234)
        first = flag_gems.randn((numel,), dtype=dtype, device=flag_gems.device)
        first_state = torch.cuda.get_rng_state()
        torch.manual_seed(1234)
        second = flag_gems.randn((numel,), dtype=dtype, device=flag_gems.device)
        second_state = torch.cuda.get_rng_state()

    assert first.shape == (numel,)
    assert first.dtype == dtype
    assert torch.isfinite(first).all().item()
    assert torch.equal(first, second)
    assert torch.equal(first_state, second_state)


@pytest.mark.randn
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("nvidia", "thead")
    or os.getenv("FLAGGEMS_TEST_LARGE_RANDN") != "1",
    reason="Opt-in large allocation; requires an otherwise idle CUDA/PPU device.",
)
@pytest.mark.parametrize("numel", [2**31 + 2048, 6_400_030_720])
def test_randn_large_output_offsets(numel):
    # Run only with the fix: the unpatched kernel can write before its allocation.
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 32 * 1024**3:
        pytest.skip("Large randn regression requires at least 32 GiB free memory.")

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1234)
        out = flag_gems.randn((numel,), dtype=torch.bfloat16, device=flag_gems.device)
    assert out.shape == (numel,)
    for start in range(0, numel, 16 * 1024**2):
        assert torch.isfinite(out[start : start + 16 * 1024**2]).all().item()

    # Cover either side of the first signed-i32 boundary, the next wrap and tail.
    starts = [0, 2**31 - 2048, 2**31, max(0, numel - 2048)]
    if numel > 2**32 + 2048:
        starts.extend([2**32 - 2048, 2**32])
    for start in starts:
        sample = out[start : start + 2048].float()
        assert sample.mean().abs().item() < 0.1
        assert (sample.std() - 1).abs().item() < 0.1
