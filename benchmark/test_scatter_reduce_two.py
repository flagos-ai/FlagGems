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

from . import consts
from .test_scatter_reduce import (
    TensorSelectBenchmark,
    gather_scatter_gbps,
    scatter_reduce_input_fn_factory,
    skip_tsingmicro,
)


@pytest.mark.scatter_reduce_two
@skip_tsingmicro
@pytest.mark.skip_native(
    vendors=["ascend", "enflame"],
    reason="Native aten::scatter_reduce.two has no NPU kernel timing for cases "
    "such as FP16 sum on Ascend and aborts on int64 indices on Enflame",
)
def test_scatter_reduce():
    """Benchmark aten::scatter_reduce.two for every supported reduction mode."""
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce_two",
        torch_op=torch.ops.aten.scatter_reduce.two,
        input_fn=scatter_reduce_input_fn_factory(),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="Enflame does not support this scatter_reduce benchmark",
)
@skip_tsingmicro
@pytest.mark.skip_native(
    vendors="ascend",
    reason="Native aten::scatter_reduce_.two has no NPU kernel timing for cases "
    "such as FP32 prod",
)
def test_scatter_reduce_():
    """Benchmark aten::scatter_reduce_.two for every supported reduction mode."""
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce_two_",
        torch_op=torch.ops.aten.scatter_reduce_.two,
        input_fn=scatter_reduce_input_fn_factory(is_inplace=True),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="Enflame does not support this scatter_reduce benchmark",
)
@skip_tsingmicro
@pytest.mark.skip_native(
    vendors="ascend",
    reason="Native aten::scatter_reduce.two_out falls back to CPU on older runtimes "
    "and rejects include_self=False on 910B1 with CANN 9.2.1",
)
def test_scatter_reduce_out():
    """Benchmark aten::scatter_reduce.two_out for every supported reduction mode."""
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce_two_out",
        torch_op=torch.ops.aten.scatter_reduce.two_out,
        input_fn=scatter_reduce_input_fn_factory(use_out=True),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
