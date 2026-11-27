# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Check the case interface and backend shapes of the synced benchmarks."""

from pathlib import Path

import pytest
import torch
import yaml

from benchmark import base, consts
from benchmark import test_batch_norm_elemt as batch_norm
from benchmark import test_embedding_spmdm as embedding
from benchmark import test_moe_load_balance_loss as moe
from benchmark import test_softmax as softmax

ROOT = Path(__file__).resolve().parents[2]


def _benchmarks():
    benches = [
        (
            "softmax_backward_data",
            base.GenericBenchmark(
                "softmax_backward_data",
                torch_op=lambda *args: None,
                case_fn=softmax.softmax_backward_data_case_fn,
                build_inputs_fn=softmax.softmax_backward_data_build_inputs_fn,
                dtypes=[torch.float32],
            ),
            (2, 4),
        ),
        (
            "batch_norm_elemt",
            batch_norm.BatchNormElemtBenchmark(
                "batch_norm_elemt",
                torch_op=lambda *args: None,
                case_fn=batch_norm.batch_norm_elemt_case_fn,
                build_inputs_fn=batch_norm.batch_norm_elemt_build_inputs_fn,
                dtypes=[torch.float32],
            ),
            (2, 3, 4, 5),
        ),
    ]
    for weighted in (False, True):
        name = "embedding_spmdm_weighted" if weighted else "embedding_spmdm"
        benches.append(
            (
                name,
                embedding.EmbeddingSpmdmBenchmark(
                    name,
                    torch_op=lambda *args: None,
                    case_fn=lambda shape, dtype, w=weighted: embedding._case_fn(
                        shape, dtype, w
                    ),
                    build_inputs_fn=embedding._build_inputs_fn,
                    dtypes=[torch.float32],
                ),
                (11, 8, 3, 4),
            )
        )
    for masked in (False, True):
        name = "moe_load_balance_loss_mask" if masked else "moe_load_balance_loss"
        benches.append(
            (
                name,
                moe.MoeLoadBalanceLossBenchmark(
                    name,
                    torch_op=lambda *args: None,
                    masked=masked,
                    dtypes=[torch.float32],
                ),
                (8, 4, 2),
            )
        )
    return benches


@pytest.mark.parametrize("vendor", ["kunlunxin", "enflame"])
def test_backend_shapes_and_case_builders(vendor, monkeypatch):
    monkeypatch.setattr(base, "vendor_name", vendor)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)
    monkeypatch.setattr(base.Config, "query", False)
    monkeypatch.setattr(base.Config, "current_nodeid", "test::benchmark")
    monkeypatch.setattr(base.Config, "skipped_case_ids", set())
    shared = yaml.safe_load((ROOT / "benchmark/core_shapes.yaml").read_text())
    path = ROOT / f"src/flag_gems/runtime/backend/_{vendor}/core_shapes.yaml"

    for name, bench, small in _benchmarks():
        bench.device = "cpu"
        bench.set_shapes(str(path))
        assert bench.shapes == [tuple(shape) for shape in shared[name]["shapes"]]
        assert bench.supports_cases()
        cases = bench.list_cases(initialize=False).cases
        assert len(cases) == len(bench.shapes)
        assert len({case.case_id for case in cases}) == len(cases)

        bench.shapes = [small]
        case = bench.list_cases(initialize=False).cases[0]
        args = bench.build_inputs(case)
        if name == "softmax_backward_data":
            assert args[0].shape == args[1].shape == torch.Size(small)
            assert args[2:] == (1, torch.float32)
        elif name == "batch_norm_elemt":
            assert args[0].shape == torch.Size(small)
            assert all(arg.shape == torch.Size([small[1]]) for arg in args[1:5])
        elif name.startswith("embedding"):
            assert args[0].shape == torch.Size([11, 8])
            assert args[1].shape == torch.Size([3, 4])
            assert (args[2] is not None) == name.endswith("_weighted")
        else:
            assert args[0].shape == torch.Size([8, 4])
            assert args[1] == 2
            assert (args[2] is not None) == name.endswith("_mask")
