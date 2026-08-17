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

import logging

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

logger = logging.getLogger(__name__)


@pytest.mark.adaptive_max_pool1d
@pytest.mark.parametrize("shape", [(2, 3, 16), (4, 8, 32), (1, 16, 64)])
@pytest.mark.parametrize("output_size", [8, 4, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_adaptive_max_pool1d_accuracy(shape, output_size, dtype, caplog):
    """Test adaptive_max_pool1d accuracy."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Reference output
    with caplog.at_level(logging.DEBUG):
        ref_out, ref_indices = torch.nn.functional.adaptive_max_pool1d(
            ref_inp, output_size, return_indices=True
        )

    # FlagGems output - direct dispatch test
    with caplog.at_level(logging.DEBUG):
        with flag_gems.use_gems():
            res_out, res_indices = torch.ops.aten.adaptive_max_pool1d(inp, output_size)

    # Verify dispatch
    assert "GEMS ADAPTIVE_MAX_POOL1D" in caplog.text

    # Verify output values
    utils.gems_assert_close(res_out, ref_out, dtype)

    # Verify indices (exact match for max pooling)
    utils.gems_assert_equal(res_indices, ref_indices)


@pytest.mark.adaptive_max_pool1d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adaptive_max_pool1d_list_output_size(dtype):
    """Test adaptive_max_pool1d with list output_size."""
    inp = torch.randn(2, 4, 20, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    output_size = [10]

    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool1d(
        ref_inp, output_size, return_indices=True
    )

    with flag_gems.use_gems():
        res_out, res_indices = torch.nn.functional.adaptive_max_pool1d(
            inp, output_size, return_indices=True
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_indices, ref_indices)


@pytest.mark.adaptive_max_pool1d
def test_adaptive_max_pool1d_edge_cases():
    """Test edge cases: output_size=1 (global pooling)."""
    inp = torch.randn(1, 3, 100, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    output_size = 1

    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool1d(
        ref_inp, output_size, return_indices=True
    )

    with flag_gems.use_gems():
        res_out, res_indices = torch.nn.functional.adaptive_max_pool1d(
            inp, output_size, return_indices=True
        )

    utils.gems_assert_close(res_out, ref_out, torch.float32)
    utils.gems_assert_equal(res_indices, ref_indices)
