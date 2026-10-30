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

from . import base, consts, utils

SHAPES = [
    (4, 3, 32, 32),
    (8, 16, 28, 28),
    (2, 8, 16, 20),
]


def _fractional_backward_forward_fixture(
    inp, kernel_size, output_size, random_samples=None
):
    if flag_gems.vendor_name != "ascend":
        return flag_gems.fractional_max_pool2d(
            inp,
            kernel_size=kernel_size,
            output_size=output_size,
            _random_samples=random_samples,
        )
    # Build canonical flat indices from the actual input and original samples.
    # Ascend's native frontend also uses a CPU fallback for this forward API.
    if random_samples is None:
        random_samples = torch.rand(
            inp.shape[0], inp.shape[1], 2, dtype=torch.float32, device=inp.device
        )
    output, indices = torch.nn.functional.fractional_max_pool2d(
        inp.detach().to(device="cpu", dtype=torch.float64),
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
        _random_samples=random_samples.to(device="cpu", dtype=torch.float64),
    )
    return output.to(device=inp.device, dtype=inp.dtype), indices.to(device=inp.device)


class FractionalMaxPool2DBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES
        self.shape_desc = "N, C, H, W"


def _case_fn(shape, dtype):
    del dtype
    output_size = [shape[-2] // 2, shape[-1] // 2]
    yield base.BenchmarkCasePlan(
        shape={
            "self": list(shape),
            "grad_output": [*shape[:-2], *output_size],
            "indices": [*shape[:-2], *output_size],
        },
        params={"kernel_size": [2, 2], "output_size": output_size},
        builder_args=(tuple(shape), tuple(output_size)),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    shape, output_size = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    output, indices = _fractional_backward_forward_fixture(
        inp, kernel_size=2, output_size=output_size
    )
    grad_output = torch.randn_like(output)
    return (
        grad_output,
        inp,
        {
            "kernel_size": [2, 2],
            "output_size": list(output_size),
            "indices": indices,
        },
    )


def _torch_backward(grad_output, inp, **kwargs):
    return torch.ops.aten.fractional_max_pool2d_backward(
        grad_output,
        inp,
        kwargs["kernel_size"],
        kwargs["output_size"],
        kwargs["indices"],
    )


@pytest.mark.fractional_max_pool2d_backward
def test_fractional_max_pool2d_backward():
    bench = FractionalMaxPool2DBackwardBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="fractional_max_pool2d_backward",
        torch_op=_torch_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.fractional_max_pool2d_backward)
    bench.run()
