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

import contextlib
import importlib
from pathlib import Path

import pytest
import torch
import triton

import flag_gems
from flag_gems.utils.codegen_config_utils import CodeGenConfig as CommonCodeGenConfig
from flag_gems.utils.pointwise_dynamic import (
    pointwise_dynamic as common_pointwise_dynamic,
)

if flag_gems.vendor_name == "kunlunxin":
    pytestmark = pytest.mark.skip("Issue #2836: not working")


@pytest.mark.parametrize(
    "num_tiles,max_grid,want",
    [(1, 3, (1, 1)), (4, 3, (2, 2)), (6, 3, (3, 2)), (7, 3, (3, 3))],
)
def test_balanced_grid_partition(num_tiles, max_grid, want):
    module = importlib.import_module("flag_gems.utils.pointwise_dynamic")
    helper = getattr(module, "_balanced_grid_partition")
    assert helper(num_tiles, max_grid) == want


def test_balanced_grid_partition_rejects_nonpositive_tiles():
    module = importlib.import_module("flag_gems.utils.pointwise_dynamic")
    helper = getattr(module, "_balanced_grid_partition")

    with pytest.raises(ValueError, match="num_tiles must be positive"):
        helper(0, 3)


def test_balanced_grid_partition_rejects_nonpositive_max_grid():
    module = importlib.import_module("flag_gems.utils.pointwise_dynamic")
    helper = getattr(module, "_balanced_grid_partition")

    with pytest.raises(ValueError, match="max_grid_size must be positive"):
        helper(4, 0)


def test_codegen_config_balance_grid_defaults_off():
    config = CommonCodeGenConfig(4, (3, 1, 1), 4, True, False)
    assert config.balance_grid is False


def test_codegen_config_accepts_balance_grid_opt_in():
    config = CommonCodeGenConfig(
        4,
        (3, 1, 1),
        4,
        True,
        False,
        balance_grid=True,
    )
    assert config.balance_grid is True


def test_balanced_grid_partition_covers_each_tile_once():
    module = importlib.import_module("flag_gems.utils.pointwise_dynamic")
    helper = getattr(module, "_balanced_grid_partition")

    num_ctas, tiles_per_cta = helper(4, 3)

    tile_ids = [
        pid + round_id * num_ctas
        for round_id in range(tiles_per_cta)
        for pid in range(num_ctas)
        if pid + round_id * num_ctas < 4
    ]

    assert sorted(tile_ids) == [0, 1, 2, 3]
    assert len(tile_ids) == len(set(tile_ids))


@pytest.mark.parametrize("prefer_1d_tile", [False, True])
def test_balanced_grid_generated_wrapper_source_and_runtime_partition(
    monkeypatch,
    prefer_1d_tile,
):
    @triton.jit
    def copy_scalar(x):
        return x

    default_config = CommonCodeGenConfig(
        4,
        (3, 1, 1),
        4,
        False,
        prefer_1d_tile,
    )
    balanced_config = CommonCodeGenConfig(
        4,
        (3, 1, 1),
        4,
        False,
        prefer_1d_tile,
        balance_grid=True,
    )

    default_fn = common_pointwise_dynamic(
        copy_scalar,
        num_inputs=1,
        promotion_methods=[(0, "DEFAULT")],
        config=default_config,
    )
    balanced_fn = common_pointwise_dynamic(
        copy_scalar,
        num_inputs=1,
        promotion_methods=[(0, "DEFAULT")],
        config=balanced_config,
    )

    default_info = default_fn.get_kernel_info(1)
    balanced_info = balanced_fn.get_kernel_info(1)

    default_source = Path(default_info.file_path).read_text()
    balanced_source = Path(balanced_info.file_path).read_text()

    helper_import = (
        "from flag_gems.utils.pointwise_dynamic import _balanced_grid_partition"
    )

    assert helper_import not in default_source
    assert "num_ctas = min(3, num_tiles)" in default_source
    assert "tiles_per_cta = triton.cdiv(num_tiles, num_ctas)" in default_source

    assert helper_import in balanced_source
    assert (
        "num_ctas, tiles_per_cta = " "_balanced_grid_partition(num_tiles, 3)"
    ) in balanced_source

    observed_partitions = []

    class KernelLaunchSpy:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                observed_partitions.append((grid[0], kwargs["tiles_per_cta"]))

            return launch

    class NoopDeviceContext:
        @staticmethod
        def device(index):
            return contextlib.nullcontext()

    for fn, info in (
        (default_fn, default_info),
        (balanced_fn, balanced_info),
    ):
        wrapper = fn.instantiate(1)

        monkeypatch.setitem(
            wrapper.__globals__,
            info.kernel_name,
            KernelLaunchSpy(),
        )
        monkeypatch.setitem(
            wrapper.__globals__,
            "heuristics_for_tile_size",
            lambda *args: (4,),
        )
        monkeypatch.setitem(
            wrapper.__globals__,
            "heuristics_for_num_warps",
            lambda tile_size: 1,
        )
        monkeypatch.setitem(
            wrapper.__globals__,
            "torch_device_fn",
            NoopDeviceContext,
        )

        source = torch.empty(16)
        destination = torch.empty_like(source)

        assert wrapper(source, out0=destination) is destination

    assert observed_partitions == [(3, 2), (2, 2)]


def test_balanced_grid_cache_isolation():
    @triton.jit
    def copy_scalar(x):
        return x

    default_config = CommonCodeGenConfig(
        4,
        (3, 1, 1),
        4,
        False,
        True,
    )
    same_default_config = CommonCodeGenConfig(
        4,
        (3, 1, 1),
        4,
        False,
        True,
    )
    balanced_config = CommonCodeGenConfig(
        4,
        (3, 1, 1),
        4,
        False,
        True,
        balance_grid=True,
    )

    def make_function(config):
        return common_pointwise_dynamic(
            copy_scalar,
            num_inputs=1,
            promotion_methods=[(0, "DEFAULT")],
            config=config,
        )

    default_fn = make_function(default_config)
    same_default_fn = make_function(same_default_config)
    balanced_fn = make_function(balanced_config)

    default_info = default_fn.get_kernel_info(1)
    same_default_info = same_default_fn.get_kernel_info(1)
    balanced_info = balanced_fn.get_kernel_info(1)

    assert default_info.file_path == same_default_info.file_path
    assert default_info.file_path != balanced_info.file_path

    assert "_balanced" not in Path(default_info.file_path).stem
    assert Path(balanced_info.file_path).stem.endswith("_balanced")

    default_wrapper = default_fn.instantiate(1)

    default_config.balance_grid = True
    balanced_wrapper = default_fn.instantiate(1)

    assert balanced_wrapper is not default_wrapper
    assert len(default_fn.overloads) == 2
    assert len(default_fn._kernel_info_cache) == 2


@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon",
    reason="Cambricon uses a separate pointwise generator",
)
def test_balanced_grid_real_kernel_covers_non_power_of_two_tail():
    config = CommonCodeGenConfig(
        max_tile_size=4,
        max_grid_size=(3, 1, 1),
        max_num_warps_per_cta=4,
        prefer_block_pointer=False,
        prefer_1d_tile=True,
        balance_grid=True,
    )

    @common_pointwise_dynamic(
        num_inputs=1,
        promotion_methods=[(0, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def copy_scalar(x):
        return x

    source = torch.arange(
        15,
        dtype=torch.float32,
        device=flag_gems.device,
    )
    actual = copy_scalar(source)

    torch.testing.assert_close(actual, source)
