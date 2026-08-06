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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

im2col_module = importlib.import_module("flag_gems.ops.im2col")

# Cover 3D input, small/large 4D inputs for representative im2col testing
IM2COL_SHAPES = [(3, 8, 8), (1, 3, 16, 16), (16, 64, 64), (32, 128, 128)]
IM2COL_CONFIGS = [
    ((3, 3), (1, 1), (1, 1), (1, 1)),
    ((3, 3), (1, 1), (0, 0), (2, 2)),
    ((5, 4), (2, 2), (2, 1), (1, 2)),
    ((1, 1), (1, 1), (0, 0), (1, 1)),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.im2col
@pytest.mark.parametrize("shape", IM2COL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("kernel_size, dilation, padding, stride", IM2COL_CONFIGS)
def test_im2col(shape, dtype, kernel_size, dilation, padding, stride):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    ref_out = torch.ops.aten.im2col(ref_x, kernel_size, dilation, padding, stride)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.im2col(x, kernel_size, dilation, padding, stride)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.im2col
@pytest.mark.parametrize(
    "config",
    im2col_module.im2col_kernel.fn.configs,
    ids=lambda c: "-".join(f"{k}{v}" for k, v in sorted(c.kwargs.items())),
)
def test_im2col_every_autotune_config_writes_full_output(config):
    # The launch grid must cover the whole output for whichever tile sizes the
    # autotuner selects. A single tuning config sidesteps the benchmark sweep,
    # which otherwise hides under-coverage on the first call per shape: the
    # sweep runs every candidate (including a fully covering one) against the
    # same output buffer, so only calls after the sweep expose the winner's
    # real coverage. The second call below also exercises the cached-kernel
    # launch path.
    tuner = im2col_module.im2col_kernel.fn
    saved_configs = tuner.configs
    tuner.configs = [config]
    tuner.cache.clear()
    for kernel_cache in im2col_module.im2col_kernel.kernel_cache:
        kernel_cache.clear()
    try:
        # rows_total = 8*3*3 = 72 and L = 32*32 = 1024 are not covered by a
        # grid sized for other tile dimensions than the selected ones.
        x = torch.randn((1, 8, 32, 32), dtype=torch.float32, device=flag_gems.device)
        ref_x = utils.to_reference(x)
        ref_out = torch.ops.aten.im2col(ref_x, (3, 3), (1, 1), (1, 1), (1, 1))
        with flag_gems.use_gems():
            for _ in range(2):
                act_out = torch.ops.aten.im2col(x, (3, 3), (1, 1), (1, 1), (1, 1))
                utils.gems_assert_close(act_out, ref_out, dtype=torch.float32)
    finally:
        tuner.configs = saved_configs
        tuner.cache.clear()
        for kernel_cache in im2col_module.im2col_kernel.kernel_cache:
            kernel_cache.clear()
