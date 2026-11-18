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

from . import base, consts


class LinearBenchmark(base.BlasBenchmark):
    def get_tflops(self, op, *args, **kwargs):
        input_tensor = args[0]
        weight = args[1]
        batch = input_tensor.shape[0]
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        return batch * out_features * (2 * in_features + 1)


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    input_tensor = torch.randn([m, k], dtype=dtype, device=device)
    weight = torch.randn([n, k], dtype=dtype, device=device)
    bias = torch.randn([n], dtype=dtype, device=device)
    yield input_tensor, weight, bias


@pytest.mark.linear
def test_linear():
    bench = LinearBenchmark(
        op_name="linear",
        input_fn=_input_fn,
        torch_op=torch.nn.functional.linear,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


_POLICY_TRACES = {}
_FIRST_CALLS = []


def _policy_linear(x, weight, bias, expanded):
    import importlib
    from unittest.mock import patch

    import flag_gems

    module = importlib.import_module(flag_gems.linear.__module__)
    key = (
        "expanded" if expanded else "default",
        x.shape[0],
        *weight.shape,
        bias is not None,
    )

    class RecordEntry:
        def __init__(self, entry):
            self.entry = entry

        def __getattr__(self, name):
            return getattr(self.entry, name)

        def __getitem__(self, grid):
            launch = self.entry[grid]

            def call(*args, **kwargs):
                result = launch(*args, **kwargs)
                meta = {
                    name: result[1][name]
                    for name in (
                        "BLOCK_SIZE_M",
                        "BLOCK_SIZE_N",
                        "BLOCK_SIZE_K",
                        "num_warps",
                        "num_stages",
                    )
                }
                previous = _POLICY_TRACES.setdefault(key, meta)
                assert previous == meta
                return result

            return call

    default = module.linear_kernel
    small = module._small_fp32_entry
    selector = module._use_small_fp32_entry
    with (
        patch.object(module, "linear_kernel", RecordEntry(default)),
        patch.object(module, "_small_fp32_entry", lambda: RecordEntry(small())),
        patch.object(
            module,
            "_use_small_fp32_entry",
            selector if expanded else lambda *args: False,
        ),
    ):
        return flag_gems.linear(x, weight, bias)


def _default_linear_policy(x, weight, bias):
    return _policy_linear(x, weight, bias, False)


def _expanded_linear_policy(x, weight, bias):
    return _policy_linear(x, weight, bias, True)


class MetaXLinearPolicyBenchmark(base.Benchmark):
    """Isolate default versus expanded tuning pools without changing arithmetic."""

    def set_shapes(self, shape_file_path=None):
        pairs = [(3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096)]
        self.shapes = [
            (m, n, k, bias)
            for n, k in pairs
            for m in (1, 31, 32, 33, 128, 1024, 1025)
            for bias in (False, True)
        ]

    def get_input_iter(self, dtype):
        import time

        for m, n, k, has_bias in self.shapes:
            generator = torch.Generator(device=self.device).manual_seed(736)
            x = torch.randn(m, k, dtype=dtype, device=self.device, generator=generator)
            weight = torch.randn(
                n, k, dtype=dtype, device=self.device, generator=generator
            )
            bias = (
                torch.randn(n, dtype=dtype, device=self.device, generator=generator)
                if has_bias
                else None
            )
            first = {}
            for name, op in (
                ("default", _default_linear_policy),
                ("expanded", _expanded_linear_policy),
            ):
                torch.cuda.synchronize()
                started = time.perf_counter()
                op(x, weight, bias)
                torch.cuda.synchronize()
                first[name] = (time.perf_counter() - started) * 1000
                for _ in range(3):
                    op(x, weight, bias)
                torch.cuda.synchronize()
            # First-call wall time includes any tuning/JIT/cache lookup; never
            # mix it with steady graph replay latency or call it pure tuning time.
            _FIRST_CALLS.append(
                {"shape": [m, n, k], "bias": has_bias, "wall_ms": first}
            )
            yield x, weight, bias


@pytest.mark.linear
@pytest.mark.skipif(
    __import__("flag_gems").vendor_name != "metax", reason="MetaX Linear policy"
)
def test_metax_linear_tuning_policy():
    import json
    from pathlib import Path

    from .conftest import Config

    if Config.mode != consts.BenchMode.CUDAGRAPH:
        pytest.skip(
            "Policy comparison requires graph replay to exclude host override overhead"
        )
    _POLICY_TRACES.clear()
    _FIRST_CALLS.clear()
    bench = MetaXLinearPolicyBenchmark(
        op_name="linear",
        torch_op=_default_linear_policy,
        gems_op=_expanded_linear_policy,
        dtypes=[torch.float32],
    )
    with torch.inference_mode():
        bench.run()
    if Config.record_json:
        path = Path(str(Config.output) + ".trace.json")
        path.write_text(
            json.dumps(
                {
                    "scope": "operator policy only; first-call wall times are not steady latency",
                    "first_calls": _FIRST_CALLS,
                    "launches": [
                        {"key": key, "config": config}
                        for key, config in _POLICY_TRACES.items()
                    ],
                },
                indent=2,
            )
        )
