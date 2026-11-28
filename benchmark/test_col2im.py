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


class Col2ImBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (2, 3, (2, 2), (4, 5), (1, 1), (0, 0), (1, 1)),
        (2, 16, (3, 3), (16, 16), (1, 1), (0, 0), (1, 1)),
        (4, 32, (3, 3), (32, 32), (1, 1), (1, 1), (1, 1)),
        (2, 64, (3, 3), (64, 64), (2, 2), (1, 1), (1, 1)),
        (1, 128, (3, 3), (128, 128), (1, 1), (1, 1), (1, 1)),
        (2, 64, (5, 5), (64, 64), (2, 2), (2, 2), (1, 1)),
        (4, 32, (3, 3), (128, 128), (2, 2), (1, 1), (2, 2)),
    ]

    def set_more_shapes(self):
        return []

    def set_shapes(self, *args, **kwargs):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for config in self.shapes:
            (
                batch,
                channels,
                kernel_size,
                output_size,
                stride,
                padding,
                dilation,
            ) = config
            kernel_h, kernel_w = kernel_size
            output_h, output_w = output_size
            stride_h, stride_w = stride
            padding_h, padding_w = padding
            dilation_h, dilation_w = dilation
            L_h = (
                output_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
            ) // stride_h + 1
            L_w = (
                output_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
            ) // stride_w + 1
            L = L_h * L_w
            inp = torch.randn(
                batch,
                channels * kernel_h * kernel_w,
                L,
                device=self.device,
                dtype=dtype,
            )
            yield inp, output_size, kernel_size, dilation, padding, stride


@pytest.mark.col2im
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_col2im():
    bench = Col2ImBenchmark(
        op_name="col2im",
        torch_op=torch.ops.aten.col2im,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# (col shape, output_size, kernel_size, dilation, padding, stride)
_COL2IM_BACKWARD_CASES = [
    ((2, 9, 4), (4, 4), (3, 3), (1, 1), (0, 0), (1, 1)),
    ((4, 9, 1024), (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)),
    ((2, 16, 3844), (128, 128), (4, 4), (2, 2), (1, 1), (2, 2)),
    ((1, 9, 225), (32, 32), (3, 3), (1, 1), (0, 0), (2, 2)),
]


def col2im_backward_case_fn(case, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(case[0])},
        params={},
        builder_args=(case,),
    )


def col2im_backward_build_inputs_fn(plan, dtype, device):
    (case,) = plan.builder_args
    shape, *args = case
    # ``self`` is the column tensor; ``grad_output`` carries the forward
    # output's shape, so derive it from the forward itself.
    inp = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)
    grad_output = torch.randn(
        torch.ops.aten.col2im(inp.detach(), *args).shape, dtype=dtype, device=device
    )
    return grad_output, inp, *args


def col2im_backward_torch_op(grad_output, inp, *args):
    inp = inp.detach().requires_grad_(True)
    (ref_in_grad,) = torch.autograd.grad(
        torch.ops.aten.col2im(inp, *args), inp, grad_output
    )
    return ref_in_grad


class Col2ImBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _COL2IM_BACKWARD_CASES
        self.shape_desc = "(col shape, out, kernel, dilation, pad, stride)"

    def set_more_shapes(self):
        return []


@pytest.mark.col2im_backward
def test_col2im_backward():
    bench = Col2ImBackwardBenchmark(
        case_fn=col2im_backward_case_fn,
        build_inputs_fn=col2im_backward_build_inputs_fn,
        op_name="col2im_backward",
        torch_op=col2im_backward_torch_op,
        gems_op=flag_gems.col2im_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
