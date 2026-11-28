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

from . import base


class CdistForwardBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # (batch, n1, dim) — mirrors the sibling cdist_backward benchmark; the
        # (16, 128, 256) case is the large shape where the p=2 GEMM path pays off.
        self.shapes = [(2, 16, 32), (4, 32, 64), (8, 64, 128), (16, 128, 256)]

    def get_input_iter(self, cur_dtype):
        for batch, n1, dim in self.shapes:
            n2 = n1 // 2 + 1
            x1 = torch.randn(batch, n1, dim, dtype=cur_dtype, device=self.device)
            x2 = torch.randn(batch, n2, dim, dtype=cur_dtype, device=self.device)
            cdist = torch.cdist(x1, x2, p=2.0)
            grad = torch.randn(batch, n1, n2, dtype=cur_dtype, device=self.device)
            yield grad, x1, x2, 2.0, cdist

    def get_gbps(self, args, latency=None):
        # Each output element is read/written twice (read xk, write grad);
        # the reduced axis r is streamed once.
        grad, x1, x2 = args[0], args[1], args[2]
        n = (grad.numel() + x1.numel() + x2.numel()) * grad.element_size()
        return n / (latency * 1e-3) / 1e9

    def get_tflops(self, op, *args, **kwargs):
        # Pure data-movement / fused-multiply-accumulate restore; no
        # dedicated matmul-like FLOP model.
        return 0.0


@pytest.mark.cdist_forward_backward
def test_cdist_forward_backward():
    bench = CdistForwardBackwardBenchmark(
        op_name="cdist_forward_backward",
        torch_op=_aten_reference,
        # _cdist_backward uses fp32 accumulation; only float32 is numerically stable
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.cdist_forward_backward)
    bench.run()


def _aten_reference(grad, x1, x2, p, cdist):
    ref1 = torch.ops.aten._cdist_backward(grad, x1, x2, p, cdist)
    ref2 = torch.ops.aten._cdist_backward(
        grad.transpose(-2, -1), x2, x1, p, cdist.transpose(-2, -1)
    )
    return ref1, ref2
