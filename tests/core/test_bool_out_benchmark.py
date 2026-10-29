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

import pytest
import torch

from benchmark import base, conftest


@pytest.mark.parametrize("native", [torch.isneginf, torch.signbit])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bool_out_input_generation_and_case_identity(monkeypatch, native, dtype):
    config = conftest.BenchConfig()
    config.current_nodeid = "benchmark/test_example.py::test_example"
    monkeypatch.setattr(base, "Config", config)
    ordinary = base.UnaryPointwiseOutBenchmark("example", native, dtypes=[dtype])
    boolean = base.BoolUnaryPointwiseOutBenchmark("example", native, dtypes=[dtype])
    for bench in (ordinary, boolean):
        bench.shapes = [(2, 3), (0,), ()]
        bench.device = "cpu"
        assert bench.supports_cases()
    ordinary_cases = ordinary.list_cases(initialize=False).cases
    boolean_cases = boolean.list_cases(initialize=False).cases
    assert [case.case_id for case in ordinary_cases] == [
        case.case_id for case in boolean_cases
    ]
    for bench, cases, output_dtype in (
        (ordinary, ordinary_cases, dtype),
        (boolean, boolean_cases, torch.bool),
    ):
        inputs = [bench.build_inputs(case) for case in cases]
        inputs += list(bench.get_input_iter(dtype))
        for inp, kwargs in inputs:
            out = kwargs["out"]
            assert inp.dtype == dtype
            assert out.dtype == output_dtype
            assert out.shape == inp.shape
            if bench is boolean:
                assert native(inp, out=out) is out


@pytest.mark.parametrize("native", [torch.isneginf, torch.signbit])
def test_reference_only_uses_bool_out(monkeypatch, native):
    config = conftest.BenchConfig()
    config.reference_only = True
    config.current_nodeid = "benchmark/test_example.py::test_example"
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(base.torch_device_fn, "synchronize", lambda: None)
    bench = base.BoolUnaryPointwiseOutBenchmark(
        "example", native, dtypes=[torch.float32]
    )
    bench.device = "cpu"
    bench.shapes = [(2, 3)]
    case = bench.list_cases(initialize=False).cases[0]

    def time_reference(op, *args, **kwargs):
        op(*args, **kwargs)
        return 1.0

    monkeypatch.setattr(bench, "get_latency", time_reference)
    assert bench._run_reference_cases([case.case_id]) == [case.case_id]
    assert config.reference_records[0]["status"] == "PASSED"
