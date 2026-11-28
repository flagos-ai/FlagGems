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

"""Performance tests for ``foreach_minimum_backward``.

There is no ``aten::_foreach_minimum_backward`` to dispatch to, so the ATen
baseline is the autograd chain the derivatives.yaml formula compiles to
(``where``, ``div``, ``masked_fill`` per tensor) while the candidate is the
fused single-launch Triton kernel driven through the explicit
``foreach_minimum_backward`` entry point.
"""

from typing import Generator

import pytest
import torch

from flag_gems.ops.foreach_minimum_backward import foreach_minimum_backward

from . import base, consts


def _aten_reference(grad_output, self_t, scalar, scalars):
    """ATen's derivatives.yaml formula, the three-kernel chain per tensor."""
    outs = []
    for i, (g, t) in enumerate(zip(grad_output, self_t)):
        s = scalar if scalar is not None else scalars[i]
        r = torch.where(t == s, g / 2, g)
        outs.append(r.masked_fill(t > s, 0))
    return outs


class ForeachMinimumBackwardBenchmark(base.Benchmark):
    """16-tensor lists over the shared foreach shape policy."""

    DEFAULT_SHAPES = [(256, 256), (512, 512), (1024, 1024)]
    DEFAULT_SHAPE_DESC = "M, N (per tensor, list of 16)"
    LIST_LENGTH = 16

    def set_more_shapes(self):
        return [(2**i,) for i in range(10, 20, 4)]

    def get_input_iter(self, dtype) -> Generator:
        n = self.LIST_LENGTH
        for shape in self.shapes:
            selfs = [
                torch.randn(shape, device=self.device, dtype=dtype) for _ in range(n)
            ]
            grads = [torch.randn_like(t) for t in selfs]
            if self.overload == "Scalar":
                yield (grads, selfs, 0.35, None)
            else:
                yield (grads, selfs, None, [float(i % 5) - 2.0 for i in range(n)])

    def run(self, case_ids=None):
        results = []
        for dtype in consts.FLOAT_DTYPES:
            for grads, selfs, scalar, scalars in self.get_input_iter(dtype):
                results.append(self._measure_pair(grads, selfs, scalar, scalars, dtype))
        return results

    def _measure_pair(self, grads, selfs, scalar, scalars, dtype):
        import triton

        def aten_ref():
            return _aten_reference(grads, selfs, scalar, scalars)

        def gems():
            return foreach_minimum_backward(
                grads, selfs, scalar=scalar, scalars=scalars
            )

        latency_base = triton.testing.do_bench(
            aten_ref, warmup=25, rep=100, return_mode="median"
        )
        latency = triton.testing.do_bench(
            gems, warmup=25, rep=100, return_mode="median"
        )
        speedup = (latency_base + 1e-7) / (latency + 1e-7)
        overload = "Scalar" if scalar is not None else "ScalarList"
        print(
            f"foreach_minimum_backward[{overload}] {dtype} "
            f"{tuple(selfs[0].shape)}x{len(selfs)}: "
            f"aten {latency_base:.4f}ms -> gems {latency:.4f}ms ({speedup:.2f}x)"
        )
        return {"latency_base": latency_base, "latency": latency, "speedup": speedup}


@pytest.mark.foreach_minimum_backward
def test_foreach_minimum_backward_scalar():
    bench = ForeachMinimumBackwardBenchmark(
        op_name="foreach_minimum_backward_scalar", torch_op=None
    )
    bench.overload = "Scalar"
    bench.run()


@pytest.mark.foreach_minimum_backward
def test_foreach_minimum_backward_scalar_list():
    bench = ForeachMinimumBackwardBenchmark(
        op_name="foreach_minimum_backward_scalar_list", torch_op=None
    )
    bench.overload = "ScalarList"
    bench.run()
