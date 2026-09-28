"""Check that a packaged model only receives candidates it can score."""

from types import SimpleNamespace

import pytest


def test_run_proposer_filters_runtime_configs_outside_model_domain():
    triton = pytest.importorskip("triton")
    from flag_gems.flagtune.inference import cost_model

    seen = []
    variant = SimpleNamespace(
        param_names=("BLOCK_M",),
        param_space=SimpleNamespace(validate=lambda config: config["BLOCK_M"] <= 64),
        to_config=lambda config: triton.Config(
            {"BLOCK_M": config["BLOCK_M"]}, num_warps=4, num_stages=2
        ),
    )
    identity = SimpleNamespace(
        artifact_key="thead-zw810e/flaggems/mm/gemv_ppu/bf16-bf16-bf16",
        op_id="flaggems/mm",
        variant="gemv_ppu",
        platform_key="thead-zw810e",
        dtype_key="bf16-bf16-bf16",
    )

    def propose(_benchmark, _arguments, initial, _meta):
        seen.extend(initial)
        return [{"BLOCK_M": 64}]

    candidates = [
        triton.Config({"BLOCK_M": block_m}, num_warps=4, num_stages=2)
        for block_m in (64, 128)
    ]
    selected, timings = cost_model.run_proposer(
        SimpleNamespace(), lambda _config: [0.01], candidates, {},
        (identity, propose, variant),
    )

    assert [config["BLOCK_M"] for config in seen] == [64]
    assert selected.kwargs["BLOCK_M"] == 64
    assert list(timings.values()) == [0.01]

    with pytest.raises(ValueError, match="no runtime candidates satisfy"):
        cost_model.run_proposer(
            SimpleNamespace(), lambda _config: [0.01], candidates[1:], {},
            (identity, propose, variant),
        )
