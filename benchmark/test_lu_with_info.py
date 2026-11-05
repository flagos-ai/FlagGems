# Copyright 2026, The FlagOS Contributors.
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

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

# ``torch._lu_with_info`` is only well-supported with double precision on the
# accelerator in the native baseline; restrict the benchmark dtype set per
# vendor (float64 is exercised on nvidia, float32 elsewhere). Half precision
# is not supported by the LU factorization kernels.
if VENDOR == "nvidia":
    # CUDA LU only exposes float32/float64 baselines; half precision unsupported.
    _TEST_DTYPES = [torch.float32, torch.float64]
else:
    _TEST_DTYPES = [torch.float32]


class LuWithInfoBenchmark(base.Benchmark):
    """Benchmark for ``torch._lu_with_info`` (LU with pivoting + info)."""

    DEFAULT_SHAPE_DESC = "input shape"
    DEFAULT_DTYPES = _TEST_DTYPES
    # Square and batched-square inputs; the (8, 128, 128) case exercises the
    # batched LU path. Sizes chosen to keep the per-case runtime bounded while
    # spanning the kernel's tiling regimes.
    DEFAULT_SHAPES = (
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (8, 128, 128),
    )

    def set_shapes(self, shape_file_path=None):
        # Override and ignore core_shapes.yaml: this is a non-pointwise LU op
        # whose shapes are 2-D/3-D matrices. The generic ``Benchmark:`` entry
        # in core_shapes.yaml carries 1-D sizes (e.g. [1073741824]) that would
        # crash the LU kernel ("Expected tensor with 2 or more dimensions"), so
        # we pin the benchmark to this op's own matrix shapes unconditionally.
        self.shapes = list(self.DEFAULT_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
            yield inp, {}


@pytest.mark.lu_with_info
def test_lu_with_info():
    bench = LuWithInfoBenchmark(
        op_name="lu_with_info",
        torch_op=torch._lu_with_info,
        gems_op=flag_gems._lu_with_info,
        dtypes=_TEST_DTYPES,
    )
    bench.run()
