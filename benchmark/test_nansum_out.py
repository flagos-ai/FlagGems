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

from . import base, consts

DTYPES = consts.FLOAT_DTYPES + [torch.int8, torch.uint8]


class _NansumOutBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            if cur_dtype.is_floating_point:
                x = torch.randn(shape, dtype=cur_dtype, device=self.device) * 10
                mask = torch.rand(shape, device=self.device) > 0.7
                x[mask] = float("nan")
            else:
                # integers hold no NaN; no int8/uint8 randint on device
                info = torch.iinfo(cur_dtype)
                x = torch.randint(
                    info.min, info.max, shape, dtype=torch.int64, device="cpu"
                ).to(self.device, cur_dtype)

            out = torch.empty((), dtype=cur_dtype, device=self.device)
            yield (x, {"out": out})

            if x.ndim >= 2:
                yield x, -1, {
                    "out": torch.empty(
                        x.shape[:-1], dtype=cur_dtype, device=self.device
                    )
                }
                yield x, 0, {
                    "out": torch.empty(x.shape[1:], dtype=cur_dtype, device=self.device)
                }


def _nansum_out_baseline(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    # The aten baseline writes into the caller's out via the .out overload.
    result = torch.nansum(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


@pytest.mark.nansum_out
def test_benchmark_nansum_out():
    bench = _NansumOutBenchmark(
        op_name="nansum_out",
        torch_op=_nansum_out_baseline,
        dtypes=DTYPES,
    )
    bench.set_gems(flag_gems.nansum_out)
    bench.run()
