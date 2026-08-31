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

from benchmark.consts import BenchmarkMetrics, BenchmarkResult


def test_benchmark_result_str_with_empty_results():
    """BenchmarkResult.__str__ must not crash when no metrics were collected.

    Regression test for https://github.com/flagos-ai/FlagGems/issues/2176.
    When an input generator yields nothing (e.g. a benchmark level mismatch),
    ``result`` is an empty list and printing the result previously raised
    ``IndexError: list index out of range``.
    """
    result = BenchmarkResult(
        op_name="nll_loss_nd",
        dtype="torch.float16",
        mode="kernel",
        level="core",
        result=[],
    )
    text = str(result)
    assert "No benchmark results were collected." in text


def test_benchmark_result_str_with_non_empty_results():
    """Non-empty benchmark results must still be formatted as before."""
    metrics = BenchmarkMetrics(
        shape_detail=[[64, 64], [64]],
        latency_base=0.01,
        latency=0.008,
        speedup=1.25,
    )
    result = BenchmarkResult(
        op_name="nll_loss_nd",
        dtype="torch.float16",
        mode="kernel",
        level="core",
        result=[metrics],
    )
    text = str(result)
    assert "SUCCESS" in text
    assert "0.010000" in text
    assert "1.250" in text
