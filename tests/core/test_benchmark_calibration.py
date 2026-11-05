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

"""Calibration must estimate steady-state calls independently of cold startup."""

from types import SimpleNamespace

import pytest

from benchmark import base, conftest, consts


@pytest.mark.parametrize("mode", [consts.BenchMode.OPERATOR, consts.BenchMode.WRAPPER])
@pytest.mark.parametrize("startup_seconds", [1, 100])
def test_calibration_excludes_startup(monkeypatch, mode, startup_seconds):
    config = conftest.BenchConfig()
    config.mode = mode
    config.warm_up = 1000
    config.repetition = 100
    clock = {"host": 0.0, "pending": 0.0, "cold": True}

    def synchronize():
        clock["host"] += clock["pending"]
        clock["pending"] = 0.0

    def operation():
        if clock["cold"]:
            clock["host"] += startup_seconds
            clock["pending"] += startup_seconds
            clock["cold"] = False
        else:
            clock["host"] += 1 / 1024
            clock["pending"] += 2 / 1024

    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(base, "time", SimpleNamespace(time=lambda: clock["host"]))
    monkeypatch.setattr(
        base, "torch_device_fn", SimpleNamespace(synchronize=synchronize)
    )
    expected = (341, 34) if mode == consts.BenchMode.OPERATOR else (1024, 102)
    assert base.get_iter_count(operation) == expected
