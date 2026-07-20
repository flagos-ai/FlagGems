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

from . import base, consts

# Benchmark for _resize_output
_RESIZE_OUTPUT_SHAPES = [
    (1024,),
    (2048,),
    (4096,),
    (1024, 1024),
    (2048, 512),
    (512, 2048),
]


class ResizeOutputBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return _RESIZE_OUTPUT_SHAPES


def resize_output_input_fn(shape, dtype, device):
    # Create input tensor
    inp = torch.randn(*shape, device=device, dtype=dtype)
    # Target size - same number of elements but different shape when possible
    numel = inp.numel()
    if numel == 1024:
        target_size = [32, 32]
    elif numel == 2048:
        target_size = [64, 32]
    elif numel == 4096:
        target_size = [64, 64]
    elif numel == 1024 * 1024:
        target_size = [1024, 1024]
    elif numel == 2048 * 512:
        target_size = [2048, 512]
    elif numel == 512 * 2048:
        target_size = [512, 2048]
    else:
        target_size = [numel]
    yield inp, target_size, {"device": device}


@pytest.mark.resize_output
def test_resize_output():
    # Note: PyTorch doesn't have _resize_output implemented for CUDA, so we use
    # a dummy torch_op that just calls our gems implementation as the "baseline"
    # This is because the operator is not available in PyTorch for CUDA backend

    # Create a wrapper that uses the same implementation as gems (for fair comparison)
    from flag_gems import _resize_output as gems_resize_output

    def dummy_torch_op(inp, size, device):
        # Use the same implementation as GEMS for baseline
        return gems_resize_output(inp, size, device)

    bench = ResizeOutputBenchmark(
        input_fn=resize_output_input_fn,
        op_name="resize_output",
        torch_op=dummy_torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
