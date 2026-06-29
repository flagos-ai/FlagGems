import importlib
from pathlib import Path

import triton


def _find_flagtune_tuner(kernel):
    fn = getattr(kernel, "fn", None)
    while fn is not None:
        if getattr(fn, "_flagtune_op_name", None) == "mul":
            return fn
        fn = getattr(fn, "fn", None)
    return None


def _config_signature(config):
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def test_mul_is_registered_for_flagtune():
    flagtune_module = importlib.import_module("flag_gems.runtime.flagtune")

    assert "mul" in flagtune_module.get_supported_flagtune_ops()


def test_mul_kernel_has_flagtune_tuner():
    mul_module = importlib.import_module(
        "flag_gems.runtime.backend._nvidia.hopper.ops.mul"
    )

    kernel = mul_module.mul_kernel
    tuner = _find_flagtune_tuner(kernel)

    assert getattr(kernel, "_has_flagtune_tuner", False) is True
    assert tuner is not None
    assert tuner._flagtune_expand_op_name == "mul"
    assert tuner._flagtune_yaml_path is None
    assert "dtype" in kernel.arg_names


def test_mul_default_configs_are_preserved():
    mul_module = importlib.import_module(
        "flag_gems.runtime.backend._nvidia.hopper.ops.mul"
    )
    tuner = _find_flagtune_tuner(mul_module.mul_kernel)

    assert tuner is not None
    assert [_config_signature(config) for config in tuner._flagtune_default_configs] == [
        {
            "kwargs": {"BLOCK_SIZE": 1024},
            "num_warps": 4,
            "num_stages": 3,
        },
    ]


def test_mul_common_key_order_and_strategy():
    common = importlib.import_module("flag_gems.runtime.common")

    assert common.OP_KEY_ORDERS["mul"] == ["n_elements", "dtype"]
    assert common.DEFAULT_STRATEGIES["mul"] == ["align32", "default"]


def test_mul_expand_config_path_is_discovered_by_loader(monkeypatch):
    configs_loader = importlib.import_module("flag_gems.runtime.configs_loader")
    loader = configs_loader.TunedConfigLoader()

    def fake_expand_config_path(op_name):
        if op_name == "mul":
            return "/tmp/mul_hopper_expand.yaml"
        return None

    monkeypatch.setattr(loader, "_get_expand_config_path", fake_expand_config_path)

    registry = loader._build_expand_registry()
    spec = registry["mul"]

    assert spec["expand_yaml_path"] == "/tmp/mul_hopper_expand.yaml"
    assert spec["key"] == ["n_elements", "dtype"]
    assert spec["default_strategy"] == ["align32", "default"]


def test_mul_hopper_yaml_expands_to_expected_configs():
    runtime_module = importlib.import_module("flag_gems.runtime")
    yaml_path = Path(
        "src/flag_gems/runtime/backend/_nvidia/hopper/mul_hopper_expand.yaml"
    )

    configs = runtime_module.ops_get_configs("mul", yaml_path=str(yaml_path))
    signatures = [_config_signature(config) for config in configs]

    assert len(configs) == 20
    assert {
        "kwargs": {"BLOCK_SIZE": 1024},
        "num_warps": 4,
        "num_stages": 3,
    } in signatures
    assert {
        "kwargs": {"BLOCK_SIZE": 2048},
        "num_warps": 8,
        "num_stages": 4,
    } in signatures


def test_mul_flagtune_switches_default_to_expand_configs(monkeypatch):
    flagtune_module = importlib.import_module("flag_gems.runtime.flagtune")
    runtime_module = importlib.import_module("flag_gems.runtime")
    mul_module = importlib.import_module(
        "flag_gems.runtime.backend._nvidia.hopper.ops.mul"
    )

    tuner = _find_flagtune_tuner(mul_module.mul_kernel)
    assert tuner is not None

    monkeypatch.delenv(flagtune_module.USE_FLAGTUNE_ENV, raising=False)
    monkeypatch.delenv(flagtune_module.FLAGTUNE_INCLUDE_ENV, raising=False)
    monkeypatch.setattr(flagtune_module, "_include_ops", None)
    tuner._set_configs_and_strategy(
        tuner._flagtune_default_configs,
        tuner._flagtune_default_strategy,
    )
    tuner._flagtune_active = False

    default_configs = list(tuner.configs)
    assert [_config_signature(config) for config in default_configs] == [
        {
            "kwargs": {"BLOCK_SIZE": 1024},
            "num_warps": 4,
            "num_stages": 3,
        },
    ]

    assert flagtune_module.flagtune_enabled("mul") is False
    assert tuner.apply_flagtune() is False
    assert tuner._flagtune_active is False
    assert tuner.configs == default_configs

    expand_configs = [
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
    ]
    expand_config = {
        "ranges": {
            "BLOCK_SIZE": [512, 1024, 2048],
            "s": [3, 4],
            "w": [4, 8],
        },
        "strategy": ["align32", "default"],
    }

    def fake_get_expand_config(op_name, yaml_path=None):
        assert op_name == "mul"
        assert yaml_path is None
        return expand_config

    def fake_ops_get_configs(op_name, pre_hook=None, yaml_path=None):
        assert op_name == "mul"
        assert pre_hook is None
        assert yaml_path is None
        return expand_configs

    monkeypatch.setattr(runtime_module, "get_expand_config", fake_get_expand_config)
    monkeypatch.setattr(runtime_module, "ops_get_configs", fake_ops_get_configs)

    flagtune_module.flagtune(include="mul")
    assert flagtune_module.flagtune_enabled("mul") is True
    assert tuner.apply_flagtune() is True
    assert tuner._flagtune_active is True
    assert tuner.configs == expand_configs
    assert len(tuner.configs) > len(default_configs)
    assert [_config_signature(config) for config in tuner.configs] == [
        {
            "kwargs": {"BLOCK_SIZE": 512},
            "num_warps": 4,
            "num_stages": 3,
        },
        {
            "kwargs": {"BLOCK_SIZE": 1024},
            "num_warps": 4,
            "num_stages": 3,
        },
        {
            "kwargs": {"BLOCK_SIZE": 2048},
            "num_warps": 8,
            "num_stages": 4,
        },
    ]

    monkeypatch.setattr(flagtune_module, "_include_ops", frozenset())
    monkeypatch.delenv(flagtune_module.FLAGTUNE_INCLUDE_ENV, raising=False)

    assert flagtune_module.flagtune_enabled("mul") is False
    assert tuner.apply_flagtune() is True
    assert tuner._flagtune_active is False
    assert tuner.configs == default_configs
