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
#
import pytest
import torch

import flag_gems

from . import base

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name in ("mthreads", "ascend", "hygon"),
    reason=(
        "native cudnn_batch_norm has no MUSA kernel, falls back to CPU on Ascend, "
        "and is not compiled with cuDNN support on Hygon"
    ),
)

DTYPES = [torch.float16, torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


class CudnnBatchNormBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (16, 32),
        (16, 16, 64),
        (16, 16, 1024),
        (16, 16, 4098),
        (1, 8, 4, 4),
        (16, 8, 128, 128),
        (2, 16, 7, 9, 11),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.DEFAULT_SHAPES)
        self.shape_desc = "N, C, spatial dimensions"


def _inputs(shape, dtype, device, out=False):
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    for training in (True, False):
        x = torch.randn(shape, dtype=dtype, device=device)
        w = torch.randn(shape[1], dtype=param_dtype, device=device)
        b = torch.randn_like(w)
        rm = torch.zeros_like(w)
        rv = torch.ones_like(w)
        args = (x, w, b, rm, rv, training, 0.1, 1e-5)
        if out:
            kwargs = dict(
                out0=torch.empty_like(x),
                out1=torch.empty(
                    shape[1] if training else 0, dtype=param_dtype, device=device
                ),
                out2=torch.empty(
                    shape[1] if training else 0, dtype=param_dtype, device=device
                ),
                out3=torch.empty(0, dtype=torch.uint8, device=device),
            )
            yield (*args, kwargs)
        else:
            yield args


def _out_inputs(shape, dtype, device):
    yield from _inputs(shape, dtype, device, out=True)


@pytest.mark.cudnn_batch_norm
def test_cudnn_batch_norm():
    bench = CudnnBatchNormBenchmark(
        input_fn=_inputs,
        op_name="cudnn_batch_norm",
        torch_op=torch.ops.aten.cudnn_batch_norm.default,
        dtypes=DTYPES,
    )
    bench.set_gems(flag_gems.cudnn_batch_norm)
    bench.run()


@pytest.mark.cudnn_batch_norm_out
@pytest.mark.skip_native(
    vendors="nvidia",
    reason="Native cudnn_batch_norm.out can segfault on NVIDIA",
)
def test_cudnn_batch_norm_out():
    bench = CudnnBatchNormBenchmark(
        input_fn=_out_inputs,
        op_name="cudnn_batch_norm_out",
        torch_op=torch.ops.aten.cudnn_batch_norm.out,
        dtypes=DTYPES,
    )
    bench.set_gems(flag_gems.cudnn_batch_norm_out)
    bench.run()
