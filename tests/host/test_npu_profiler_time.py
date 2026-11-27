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

"""Run with pytest --noconftest: these tests must not import Torch or Gems."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "gems_timing_under_test",
    Path(__file__).resolve().parents[2] / "benchmark/performance_utils.py",
)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def test_multikernel_calls_are_summed_and_converted_to_milliseconds(tmp_path):
    csv = tmp_path / "kernel_details.csv"
    csv.write_text("Device_id,Name,Duration(us)\n0,copy,2\n0,gdn,280\n0,copy,4\n0,gdn,300\n")
    assert timing.device_time_ms([csv], 2) == pytest.approx(0.293)


@pytest.mark.parametrize("values", ["", "0,op,nan\n", "0,op,-1\n", "0,op,0\n", "0,op,1\n1,op,1\n"])
def test_no_invalid_or_ambiguous_timing_is_reported(tmp_path, values):
    csv = tmp_path / "kernel_details.csv"
    csv.write_text("Device_id,Name,Duration(us)\n" + values)
    with pytest.raises(RuntimeError):
        timing.device_time_ms([csv], 1)


def test_capture_is_required_and_invocation_count_is_explicit(tmp_path):
    with pytest.raises(RuntimeError):
        timing.device_time_ms([], 1)
    with pytest.raises(ValueError):
        timing.device_time_ms([], 0)


def test_capture_excludes_calibration_and_warmup(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    state = {"capturing": False, "inside": 0, "outside": 0}

    def fn():
        state["inside" if state["capturing"] else "outside"] += 1

    def handler(directory, *, async_mode=False):
        assert async_mode is False

        def write(_):
            (Path(directory) / "kernel_details.csv").write_text(
                "Device_id,Name,Duration(us)\n" + "0,copy,10\n0,compute,100\n" * state["inside"]
            )
        return write

    class Profile:
        def __init__(self, **kwargs):
            self.callback = kwargs["on_trace_ready"]

        def __enter__(self):
            state["capturing"] = True

        def __exit__(self, *args):
            state["capturing"] = False
            self.callback(self)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=SimpleNamespace(
        tensorboard_trace_handler=handler, profile=Profile,
        _ExperimentalConfig=lambda **kwargs: kwargs,
        ProfilerLevel=SimpleNamespace(Level0=0), ProfilerActivity=SimpleNamespace(NPU=1))))
    ticks = iter([0, 5])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
    result = timing.npu_profiler_time_ms(fn, warmup_ms=1000, benchmark_ms=2000, artifact_dir=tmp_path)
    assert state == {"capturing": False, "inside": 2, "outside": 7}
    assert result == pytest.approx(0.11)
