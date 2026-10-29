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

import functools
import math

import pytest
import torch
from flag_gems.ops._native_batch_norm_legit import (
    _native_batch_norm_legit,
    _native_batch_norm_legit_no_stats,
    _native_batch_norm_legit_no_stats_out,
    _native_batch_norm_legit_out,
)

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def _make_common_inputs(shape, dtype, device):
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(channels, dtype=dtype, device=device)
    bias = torch.randn(channels, dtype=dtype, device=device)
    return inp, weight, bias


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _default_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    channels = shape[1]
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    return inp, weight, bias, running_mean, running_var, True, 0.1, 1e-5


def _no_stats_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    return inp, weight, bias, True, 0.1, 1e-5


def _out_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    channels = shape[1]
    stats_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    outputs = {
        "out": torch.empty_like(inp),
        "save_mean": torch.empty(channels, dtype=stats_dtype, device=device),
        "save_invstd": torch.empty(channels, dtype=stats_dtype, device=device),
    }
    return inp, weight, bias, running_mean, running_var, True, 0.1, 1e-5, outputs


def _no_stats_out_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    channels = shape[1]
    stats_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    outputs = {
        "out": torch.empty_like(inp),
        "save_mean": torch.empty(channels, dtype=stats_dtype, device=device),
        "save_invstd": torch.empty(channels, dtype=stats_dtype, device=device),
    }
    return inp, weight, bias, True, 0.1, 1e-5, outputs


@functools.lru_cache(maxsize=None)
def _ascend_native_bn_returns_variance(device):
    # Probe on first execution; only the variance workaround is version-limited.
    import torch_npu

    probe = torch.tensor(
        [[1.0, 3.0, 5.0], [5.0, 7.0, 9.0]] * 8, dtype=torch.float32, device=device
    )
    weight = torch.ones(3, dtype=torch.float32, device=device)
    bias = torch.zeros(3, dtype=torch.float32, device=device)
    mean = torch.zeros(3, dtype=torch.float32, device=device)
    var = torch.ones(3, dtype=torch.float32, device=device)
    args = (probe, weight, bias, mean, var, True, 0.1, 1e-5)
    cpu_args = tuple(
        x.cpu().clone() if isinstance(x, torch.Tensor) else x for x in args
    )
    buffers_before = [
        (x.device, x.dtype, x.shape, x.stride(), x.data_ptr()) for x in (mean, var)
    ]
    actual = torch.ops.aten._native_batch_norm_legit.default(*args)
    assert isinstance(actual, tuple) and len(actual) == 3
    expected = torch.ops.aten._native_batch_norm_legit.default(*cpu_args)
    for index, value in enumerate(actual):
        assert isinstance(value, torch.Tensor)
        assert value.device == probe.device and value.dtype == torch.float32
        assert value.shape == (probe.shape if index == 0 else mean.shape)
    saved = actual[2][0].item()
    if math.isclose(saved, 4.0, rel_tol=1e-5, abs_tol=1e-5):
        if torch.__version__ != "2.9.0+cpu" or torch_npu.__version__ != "2.9.0.post2":
            raise RuntimeError(
                "Unvalidated Ascend batch-norm variance protocol runtime"
            )
        returns_variance = True
    elif math.isclose(saved, 1.0 / math.sqrt(4.0 + 1e-5), rel_tol=1e-5, abs_tol=1e-5):
        returns_variance = False
    else:
        raise RuntimeError(
            "Unrecognized Ascend native batch-norm saved-statistic protocol"
        )
    normalized = (
        (actual[0], actual[1], torch.rsqrt(actual[2] + 1e-5))
        if returns_variance
        else actual
    )
    for value, reference in zip(normalized, expected):
        torch.testing.assert_close(value.cpu(), reference, rtol=1.3e-6, atol=1e-5)
    for value, reference, before in zip((mean, var), cpu_args[3:5], buffers_before):
        assert (
            value.device,
            value.dtype,
            value.shape,
            value.stride(),
            value.data_ptr(),
        ) == before
        torch.testing.assert_close(value.cpu(), reference, rtol=1.3e-6, atol=1e-5)
    return returns_variance


def _native_batch_norm_reference(
    inp, weight, bias, running_mean, running_var, training, momentum, eps
):
    convert_variance = (
        training
        and inp.device.type == "npu"
        and _ascend_native_bn_returns_variance(inp.device)
    )
    result = torch.ops.aten._native_batch_norm_legit.default(
        inp, weight, bias, running_mean, running_var, training, momentum, eps
    )
    if convert_variance:
        output, save_mean, save_var = result
        return output, save_mean, torch.rsqrt(save_var + eps)
    return result


def _run_benchmark(op_name, torch_op, gems_op, build_inputs_fn):
    bench = NormBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=build_inputs_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.underscore_native_batch_norm_legit
def test__native_batch_norm_legit():
    _run_benchmark(
        "_native_batch_norm_legit",
        _native_batch_norm_reference,
        _native_batch_norm_legit,
        _default_build_inputs_fn,
    )


@pytest.mark.native_batch_norm_legit_no_stats
def test_native_batch_norm_legit_no_stats():
    _run_benchmark(
        "native_batch_norm_legit_no_stats",
        torch.ops.aten._native_batch_norm_legit.no_stats,
        _native_batch_norm_legit_no_stats,
        _no_stats_build_inputs_fn,
    )


@pytest.mark.native_batch_norm_legit_out
def test_native_batch_norm_legit_out():
    _run_benchmark(
        "native_batch_norm_legit_out",
        torch.ops.aten._native_batch_norm_legit.out,
        _native_batch_norm_legit_out,
        _out_build_inputs_fn,
    )


@pytest.mark.native_batch_norm_legit_no_stats_out
def test_native_batch_norm_legit_no_stats_out():
    _run_benchmark(
        "native_batch_norm_legit_no_stats_out",
        torch.ops.aten._native_batch_norm_legit.no_stats_out,
        _native_batch_norm_legit_no_stats_out,
        _no_stats_out_build_inputs_fn,
    )
