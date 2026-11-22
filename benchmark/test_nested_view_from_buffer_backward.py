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


def _eager_reference(grad_output, buffer, sizes, strides, offsets):
    """Eager baseline with the same signature as the FlagGems wrapper.

    Restores the buffer gradient with one ``as_strided`` copy per component
    over a zero buffer (plus the metadata ``tolist`` staging the native path
    would need anyway). The restore is one kernel launch per component, so
    the eager cost scales with the component count while the Triton kernel
    scatters with a constant number of launches.
    """
    out = torch.zeros_like(buffer)
    start = 0
    for c in range(sizes.shape[0]):
        comp_sizes = sizes[c].tolist()
        numel = 1
        for s in comp_sizes:
            numel *= s
        comp = grad_output[start : start + numel].view(comp_sizes)
        out.as_strided(comp_sizes, strides[c].tolist(), int(offsets[c])).copy_(comp)
        start += numel
    return out


# (num_components, rows_per_component, cols_per_component) configurations.
# The component count sweeps the kernel's launch-count advantage (the eager
# baseline pays one launch per component); sizes keep the packed payload in
# the 10^5 - 10^7 element range of the sibling nested backward benchmarks.
NESTED_VIEW_BACKWARD_SHAPES = [
    (2, 2048, 64),
    (8, 4096, 32),
    (64, 2048, 16),
    (256, 1024, 16),
    (1024, 256, 16),
]


class NestedViewFromBufferBackwardBenchmark(base.Benchmark):
    """Benchmark for the _nested_view_from_buffer_backward operator."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = NESTED_VIEW_BACKWARD_SHAPES
        self.shape_desc = "num_components, rows_per_component, cols_per_component"

    def get_input_iter(self, cur_dtype):
        for num_components, rows, cols in self.shapes:
            sizes = torch.full((num_components, 2), rows, dtype=torch.int64)
            sizes[:, 1] = cols
            strides = torch.tensor([[cols, 1]] * num_components, dtype=torch.int64)
            offsets = torch.tensor(
                [i * rows * cols for i in range(num_components)], dtype=torch.int64
            )
            total = num_components * rows * cols
            grad_output = torch.randn(total, dtype=cur_dtype, device=self.device)
            buffer = torch.zeros(total, dtype=cur_dtype, device=self.device)
            yield grad_output, buffer, sizes, strides, offsets

    def get_gbps(self, args, latency=None):
        # Each packed element is read once and written once; the metadata
        # traffic is negligible next to the payload.
        grad_output = args[0]
        return (
            2
            * grad_output.numel()
            * grad_output.element_size()
            / (latency * 1e-3)
            / 1e9
        )

    def get_tflops(self, op, *args, **kwargs):
        # Pure data-movement restore: no floating-point math.
        return 0.0


@pytest.mark.nested_view_from_buffer_backward
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_nested_view_from_buffer_backward(dtype):
    bench = NestedViewFromBufferBackwardBenchmark(
        op_name="nested_view_from_buffer_backward",
        torch_op=_eager_reference,
        dtypes=[dtype],
    )
    bench.set_gems(flag_gems._nested_view_from_buffer_backward)
    bench.run()
