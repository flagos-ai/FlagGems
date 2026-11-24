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
from flag_gems.utils import shape_utils

from . import base, consts

skip_tsingmicro = pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)

REDUCE_MODES = ("sum", "prod", "mean", "amax", "amin")
LEGACY_REDUCE_MODES = ("add", "multiply")
# None exercises the schema default by omitting the include_self keyword.
INCLUDE_SELF_CASES = (None, False)


def scatter_reduce_more_shapes(shapes):
    """Keep scatter_reduce comprehensive shapes bounded and two-dimensional."""
    if flag_gems.vendor_name == "kunlunxin":
        return []

    return [(1, 262145), (1, 524287)] + [
        shape for shape in shapes if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
    ]


class TensorSelectBenchmark(base.GenericBenchmark2DOnly):
    """Benchmark scatter-style ATen operators with tensor index inputs."""

    def set_more_metrics(self):
        """Add effective memory bandwidth to comprehensive benchmark runs."""
        return ["gbps"]

    def set_more_shapes(self):
        return scatter_reduce_more_shapes(super().set_more_shapes())


class ScatterReduceInplaceBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        generic_shapes = super().set_more_shapes()
        scatter_shapes = scatter_reduce_more_shapes(generic_shapes)
        return list(dict.fromkeys(scatter_shapes + generic_shapes))


def scatter_reduce_input_fn_factory(*, use_out=False, is_inplace=False, legacy=False):
    """Yield every parameter combination in one benchmark record per dtype."""
    reduce_modes = LEGACY_REDUCE_MODES if legacy else REDUCE_MODES
    include_self_cases = (None,) if legacy else INCLUDE_SELF_CASES

    def inner(shape, dtype, device):
        for reduce in reduce_modes:
            for include_self in include_self_cases:
                inp = torch.randn(shape, dtype=dtype, device=device)
                dim = -1
                index = torch.randint(
                    0, shape[dim], shape, dtype=torch.long, device=device
                )
                if is_inplace and reduce in ("add", "sum"):
                    src = torch.zeros(shape, dtype=dtype, device=device)
                elif is_inplace and reduce in ("multiply", "prod"):
                    src = torch.ones(shape, dtype=dtype, device=device)
                else:
                    src = torch.randn(shape, dtype=dtype, device=device)

                kwargs = {"reduce": reduce}
                if include_self is not None:
                    kwargs["include_self"] = include_self
                if use_out:
                    kwargs["out"] = torch.empty_like(inp)
                yield inp, dim, index, src, kwargs

    return inner


def gather_scatter_gbps(bench_fn_args, latency):
    """Estimate tensor traffic for aten scatter_reduce benchmark arguments."""
    inp, _, index, src = bench_fn_args[:4]
    io_amount = sum(shape_utils.size_in_bytes(item) for item in (inp, index, src, inp))
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.scatter_reduce
@skip_tsingmicro
@pytest.mark.skip_native(
    vendors="ascend",
    reason="Native aten::scatter.reduce lacks NPU reduction timing for the full "
    "legacy add/multiply benchmark",
)
def test_scatter_reduce_optest():
    """Benchmark the legacy tensor-source aten::scatter.reduce overload."""

    bench = TensorSelectBenchmark(
        op_name="scatter_reduce",
        torch_op=torch.ops.aten.scatter.reduce,
        gems_op=flag_gems.scatter,
        input_fn=scatter_reduce_input_fn_factory(legacy=True),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.scatter_reduce_
@skip_tsingmicro
@pytest.mark.skip_native(
    vendors="ascend",
    reason="Native aten::scatter_.reduce lacks NPU reduction timing for the full "
    "legacy add/multiply benchmark",
)
def test_scatter_reduce_legacy_inplace():
    bench = ScatterReduceInplaceBenchmark(
        op_name="scatter_reduce_",
        torch_op=torch.ops.aten.scatter_.reduce,
        gems_op=flag_gems.scatter_,
        input_fn=scatter_reduce_input_fn_factory(is_inplace=True, legacy=True),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
