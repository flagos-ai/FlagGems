# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU validation for the mHC public call contract."""

import ast
import inspect
from pathlib import Path

import pytest
import torch
import triton


def test_mhc_stats_api_and_empty_input():
    path = Path(__file__).parents[1] / "src/flag_gems/fused/dsv41/mhc.py"
    module = ast.parse(path.read_text())
    function = next(
        item
        for item in module.body
        if isinstance(item, ast.FunctionDef) and item.name == "hc_mix_stats"
    )
    namespace = {"torch": torch, "triton": triton, "_num_slices_for": lambda _: 1}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    stats = namespace[function.name]
    assert list(inspect.signature(stats).parameters) == ["x_flat", "hc_fn", "eps"]
    x = torch.empty((0, 64), dtype=torch.bfloat16)
    fn = torch.empty((24, 64), dtype=torch.float32)
    assert stats(x, fn, 1e-6).shape == (0, 24)
    with pytest.raises(TypeError, match="tuned"):
        stats(x, fn, 1e-6, tuned=False)
