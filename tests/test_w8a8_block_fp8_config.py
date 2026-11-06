# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch

from flag_gems.ops.w8a8_block_fp8_matmul import (
    _select_w8a8_block_fp8_config,
    get_w8a8_block_fp8_configs,
)


def test_hopper_block32_configs_match_only_tuned_m(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device=None: "NVIDIA H100 80GB HBM3"
    )
    get_w8a8_block_fp8_configs.cache_clear()
    expected = {
        (1792, 5120): ({144}, 4),
        (16384, 1280): ({120, 144}, 1),
        (5120, 4096): ({120, 144}, 2),
        (576, 5120): ({480, 576}, 4),
    }
    for (n, k), (rows, splits) in expected.items():
        configs = get_w8a8_block_fp8_configs(n, k, 32, 32)
        assert set(configs) == rows
        for m in rows:
            selected = _select_w8a8_block_fp8_config(configs, m, 32, 32)
            assert selected["kernel"] == "hopper"
            assert selected["exact_m"] is True
            assert selected["SWAP_AB"] is True
            assert selected["SPLIT_K"] == splits
            assert selected["BLOCK_SIZE_K"] == 32
        nearby = min(rows) - 1
        assert (
            _select_w8a8_block_fp8_config(configs, nearby, 32, 32).get(
                "kernel", "generic"
            )
            == "generic"
        )
    get_w8a8_block_fp8_configs.cache_clear()
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device=None: "NVIDIA H200"
    )
    assert get_w8a8_block_fp8_configs(1792, 5120, 32, 32)[144]["kernel"] == "hopper"
    get_w8a8_block_fp8_configs.cache_clear()


def test_legacy_configs_keep_nearest_m():
    configs = {
        32: {"BLOCK_SIZE_M": 32},
        128: {"BLOCK_SIZE_M": 128, "exact_m": False},
        144: {"BLOCK_SIZE_M": 64, "exact_m": True, "kernel": "hopper"},
    }
    assert _select_w8a8_block_fp8_config(configs, 120, 128, 128)["BLOCK_SIZE_M"] == 128
    assert _select_w8a8_block_fp8_config(configs, 143, 128, 128)["BLOCK_SIZE_M"] == 128
    assert _select_w8a8_block_fp8_config(configs, 144, 128, 128)["kernel"] == "hopper"


def test_missing_block32_shape_uses_default(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device=None: "NVIDIA H100 80GB HBM3"
    )
    get_w8a8_block_fp8_configs.cache_clear()
    configs = get_w8a8_block_fp8_configs(1024, 4096, 32, 32)
    assert configs is None
    assert (
        _select_w8a8_block_fp8_config(configs, 144, 32, 32).get("kernel", "generic")
        == "generic"
    )
    get_w8a8_block_fp8_configs.cache_clear()


def test_sm90_full_decode_matrix_is_finite_and_preserves_baseline():
    from flag_gems.ops._w8a8_block_fp8_configs import (
        _BASE_CONFIGS,
        _H100_MIDRANGE_FP8_CONFIGS,
        BLOCK_FP8_SM90_ABI,
    )

    assert BLOCK_FP8_SM90_ABI == 1
    assert len(_H100_MIDRANGE_FP8_CONFIGS) == 162
    for (m, n, k), (baseline, candidate) in _H100_MIDRANGE_FP8_CONFIGS.items():
        native = _BASE_CONFIGS.get((n, k))
        expected = (
            native[min(native, key=lambda row: abs(row - m))]
            if native
            else dict(
                BLOCK_SIZE_M=64,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
                GROUP_SIZE_M=32,
                num_warps=4,
                num_stages=3,
            )
        )
        assert baseline == expected
        assert candidate["BLOCK_SIZE_K"] == 32
        split = candidate.get("SPLIT_K", 1)
        assert split > 0 and split & (split - 1) == 0
        assert k % 32 == 0 and m > 0 and n > 0
