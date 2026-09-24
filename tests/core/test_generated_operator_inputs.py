"""Test-only workload helpers must not replace master's override mechanism."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmark import base, consts
from benchmark.generated_operator_utils import OperatorBenchmark


def test_candidate_dispatch_and_preflight_are_inherited_from_master():
    assert OperatorBenchmark._candidate_call is base.Benchmark._candidate_call
    assert OperatorBenchmark._run_preflight_cases is base.Benchmark._run_preflight_cases
    assert not hasattr(OperatorBenchmark, "_resolve_direct_gems_op")


@pytest.mark.parametrize("entry", ["", "Example:\n  shapes: [[3, 4]]\n", "custom:\n  shapes: [[5, 6]]\n"])
def test_custom_shapes_are_not_replaced_by_generic_dense_defaults(tmp_path, entry):
    class Example(OperatorBenchmark):
        pass
    shapes = tmp_path / "shapes.yaml"
    shapes.write_text("GenericBenchmark:\n  shapes: [[1000, 1000]]\n" + entry)
    bench = Example("custom", torch_op=lambda x: x, input_fn=lambda *args: iter(()))
    bench.set_shapes(str(shapes), default_shapes=[((8, 9), 4)])
    assert bench.shapes == ([(5, 6)] if entry.startswith("custom") else
                            [(3, 4)] if entry else [((8, 9), 4)])


def test_fresh_input_timing_does_not_mutate_source(monkeypatch):
    config = SimpleNamespace(mode=consts.BenchMode.OPERATOR, warm_up=0, repetition=0, override_registry=None)
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(base, "torch_device_fn", SimpleNamespace(synchronize=lambda: None))
    observed = []
    def mutate(x):
        observed.append(x.item())
        x.add_(1)
    bench = OperatorBenchmark("generated_", torch_op=mutate, gems_op=mutate,
                              fresh_inputs=True, input_fn=lambda *args: iter(()))
    source = torch.zeros(1)
    assert bench.get_latency(mutate, source) > 0
    assert bench.get_latency(mutate, source) > 0
    assert observed == [0, 0] and source.item() == 0


def test_fresh_profile_prepares_inputs_outside_capture_and_uses_live_override(monkeypatch):
    from benchmark import generated_operator_utils as helper
    active = False
    events = []
    @contextmanager
    def capture(backend, case_id):
        nonlocal active
        assert case_id == "case-0"
        active = True
        events.append("capture")
        try:
            yield
        finally:
            active = False
            events.append("stop")
    clone = helper._clone_benchmark_inputs
    def prepare(value):
        assert not active
        return clone(value)
    monkeypatch.setattr(helper, "_clone_benchmark_inputs", prepare)
    observed = []
    def candidate(value):
        observed.append(value.item())
        value.add_(1)
    config = SimpleNamespace(available_case_ids=set(), executed_case_ids=set(),
        profile_warmup=2, profile_iterations=2, profile_hook=capture,
        override_registry=SimpleNamespace(get_override=lambda _: candidate))
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(base, "torch_device_fn", SimpleNamespace(synchronize=lambda: None))
    def forbidden(*args):
        pytest.fail("must not call baseline or cached candidate")
    bench = OperatorBenchmark("generated_", torch_op=forbidden, gems_op=forbidden,
                              fresh_inputs=True, input_fn=lambda *args: iter(()))
    monkeypatch.setattr(bench, "_collect_cases", lambda: [SimpleNamespace(case_id="case-0")])
    monkeypatch.setattr(bench, "build_inputs", lambda _: (torch.zeros(1),))
    assert bench._run_profile_cases(["case-0"]) == ["case-0"]
    assert observed == [0, 0, 0, 0]
    assert events == ["capture", "stop", "capture", "stop"]
    assert config.executed_case_ids == {"case-0"}
