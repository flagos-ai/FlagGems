# Copyright 2026, The FlagOS Contributors.
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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    # Single tiny shape keeps QUICK_MODE fast; mirrors sibling lift_fresh_copy.
    LIFT_FRESH_SHAPES = [(2, 3)]
else:
    # Small-to-medium 2D shapes cover tail masking and multiple blocks; mirrors
    # the sibling lift_fresh_copy test shapes.
    LIFT_FRESH_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.lift_fresh
@pytest.mark.parametrize("shape", LIFT_FRESH_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift_fresh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift_fresh(ref_inp)
    res_out = flag_gems.lift_fresh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lift_fresh
@pytest.mark.parametrize(
    "numel,expect_kernel",
    [
        (2**31 - 1, True),  # last numel that stays inside 32-bit offsets
        (2**31 + 1, False),  # first numel past the boundary: native fallback
    ],
)
def test_lift_fresh_large_numel_offsets(numel, expect_kernel, monkeypatch):
    """The 32-bit copy kernel must not run for numel > 2**31-1 (Issue #6954).

    `pid * BLOCK_SIZE` is evaluated in int32, so above 2**31-1 elements the
    offsets wrap and the kernel reads/writes out of bounds. A spy on the kernel
    pins which path is taken without relying on the device to report the fault.
    """
    if flag_gems.device != "cuda":
        pytest.skip("requires CUDA-compatible memory accounting")
    if torch.cuda.mem_get_info()[0] < numel * 2 + 2**30:
        pytest.skip("requires ~5 GiB of free device memory")

    # `flag_gems.ops.lift_fresh` is the re-exported function, so import the
    # module explicitly to patch the kernel it looks up at call time.
    lift_fresh_mod = importlib.import_module("flag_gems.ops.lift_fresh")

    launched = []
    real_kernel = lift_fresh_mod._copy_kernel

    class ObserveKernel:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                launched.append(1)
                return real_kernel[grid](*args, **kwargs)

            return run

    monkeypatch.setattr(lift_fresh_mod, "_copy_kernel", ObserveKernel())

    inp = torch.zeros(numel, dtype=torch.uint8, device=flag_gems.device)
    inp[:4] = 7
    inp[-4:] = 9

    res = flag_gems.lift_fresh(inp)

    # A spy that never fires would silently pass, so both directions are checked.
    assert bool(launched) is expect_kernel
    assert res.shape == inp.shape and res.dtype == inp.dtype
    assert res.data_ptr() != inp.data_ptr()
    assert torch.equal(res[:4], inp[:4])
    assert torch.equal(res[-4:], inp[-4:])
    # Every element was copied, not only the sampled edges.
    assert res.sum().item() == 4 * 7 + 4 * 9
