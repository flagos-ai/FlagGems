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

import flag_gems
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


def _default_input_fn(shape, dtype, device):
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    channels = shape[1]
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    yield inp, weight, bias, running_mean, running_var, True, 0.1, 1e-5


def _no_stats_input_fn(shape, dtype, device):
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    yield inp, weight, bias, True, 0.1, 1e-5


def _out_input_fn(shape, dtype, device):
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    channels = shape[1]
    # The torch reference keeps the saved statistics in the input dtype on
    # kunlunxin; on other backends low-precision inputs still use fp32 stats.
    stats_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    if flag_gems.vendor_name == "kunlunxin":
        stats_dtype = dtype
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    outputs = {
        "out": torch.empty_like(inp),
        "save_mean": torch.empty(channels, dtype=stats_dtype, device=device),
        "save_invstd": torch.empty(channels, dtype=stats_dtype, device=device),
    }
    yield inp, weight, bias, running_mean, running_var, True, 0.1, 1e-5, outputs


def _no_stats_out_input_fn(shape, dtype, device):
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    channels = shape[1]
    stats_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    if flag_gems.vendor_name == "kunlunxin":
        stats_dtype = dtype
    outputs = {
        "out": torch.empty_like(inp),
        "save_mean": torch.empty(channels, dtype=stats_dtype, device=device),
        "save_invstd": torch.empty(channels, dtype=stats_dtype, device=device),
    }
    yield inp, weight, bias, True, 0.1, 1e-5, outputs


def _run_benchmark(op_name, torch_op, gems_op, input_fn):
    bench = NormBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.native_batch_norm_legit
def test_native_batch_norm_legit():
    _run_benchmark(
        "native_batch_norm_legit",
        torch.ops.aten._native_batch_norm_legit.default,
        _native_batch_norm_legit,
        _default_input_fn,
    )


@pytest.mark.native_batch_norm_legit_no_stats
def test_native_batch_norm_legit_no_stats():
    _run_benchmark(
        "native_batch_norm_legit_no_stats",
        torch.ops.aten._native_batch_norm_legit.no_stats,
        _native_batch_norm_legit_no_stats,
        _no_stats_input_fn,
    )


@pytest.mark.native_batch_norm_legit_out
def test_native_batch_norm_legit_out():
    _run_benchmark(
        "native_batch_norm_legit_out",
        torch.ops.aten._native_batch_norm_legit.out,
        _native_batch_norm_legit_out,
        _out_input_fn,
    )


@pytest.mark.native_batch_norm_legit_no_stats_out
def test_native_batch_norm_legit_no_stats_out():
    _run_benchmark(
        "native_batch_norm_legit_no_stats_out",
        torch.ops.aten._native_batch_norm_legit.no_stats_out,
        _native_batch_norm_legit_no_stats_out,
        _no_stats_out_input_fn,
    )


def _backward_input_fn(shape, dtype, device):
    """Forward-operand tuple used by the backward benchmark.

    Mirrors ``_default_input_fn`` (training with running statistics) but keeps
    ``running_mean``/``running_var`` out of the differentiable set: they are
    buffers that ATen rejects as ``requires_grad`` inputs.
    """
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    channels = shape[1]
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    yield inp, weight, bias, running_mean, running_var, True, 0.1, 1e-5


class _NativeBatchNormLegitBackwardBenchmark(NormBenchmark):
    """Time ``torch.autograd.grad`` of ``_native_batch_norm_legit``.

    The forward returns ``(output, save_mean, save_invstd)``, so the generic
    ``Benchmark._benchmark_callable`` cannot derive the upstream gradient from
    the raw return value; this class takes the normalized output instead and
    keeps the graph alive across repetitions.
    """

    def _benchmark_callable(self, op, *args, **kwargs):
        out = op(*args, **kwargs)[0]
        dout = torch.randn_like(out)
        xs = [a for a in args if torch.is_tensor(a) and a.requires_grad]
        return (
            lambda: torch.autograd.grad(
                (out,), xs, grad_outputs=(dout,), retain_graph=True
            ),
            xs,
        )

    def unpack_to_args_kwargs(self, input_tuple):
        args, kwargs = super().unpack_to_args_kwargs(input_tuple)
        # ``is_backward=True`` marks every floating tensor differentiable, but
        # ATen rejects ``running_mean``/``running_var`` (positions 3 and 4) as
        # ``requires_grad`` inputs; the normalized input and the affine
        # parameters keep their gradient so the timed graph matches a real one.
        for index in (3, 4):
            if index < len(args) and torch.is_tensor(args[index]):
                args[index].requires_grad_(False)
        return args, kwargs


@pytest.mark.native_batch_norm_legit_backward
def test_native_batch_norm_legit_backward():
    bench = _NativeBatchNormLegitBackwardBenchmark(
        op_name="native_batch_norm_legit",
        input_fn=_backward_input_fn,
        torch_op=torch.ops.aten._native_batch_norm_legit.default,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(_native_batch_norm_legit)
    bench.run()
