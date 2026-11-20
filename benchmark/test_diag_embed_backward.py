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


def diag_embed_backward_input_fn(shape, dtype, device):
    # shape is a 2D (rows, cols) plan; the diagonal plane lives in the last
    # two dims of grad_output, so the input is the leading shape with one
    # element per diagonal.
    rows, cols = shape
    diag_len = cols
    plane = (rows, cols, cols)
    grad_output = torch.randn(plane, dtype=dtype, device=device)
    inp = torch.randn((rows, diag_len), dtype=dtype, device=device)
    yield grad_output, inp


def _diag_embed_backward_ref(grad_output, x):
    """Eager reference with the same signature as the FlagGems wrapper.

    ``DiagEmbedBackward0`` extracts the diagonal of the upstream gradient
    at (offset, dim1, dim2); ``torch.diagonal`` is the exact formula.
    """
    return torch.diagonal(grad_output, 0, -2, -1)


class DiagEmbedBackwardBenchmark(base.GenericBenchmark2DOnly):
    def set_shapes(self, shape_file_path=None):
        # ``grad_output`` is 3-D (rows, cols, cols), so the plane size grows
        # with the cube of the last dim; cap it at 1024 so the 128 MiB-scale
        # benchmarks fit the 96 GB budget shared with the reference tensors.
        self.shapes = [
            (64, 64),
            (256, 256),
            (1024, 1024),
            (64, 4096),
        ]

    def set_more_shapes(self):
        return []


@pytest.mark.diag_embed_backward
def test_diag_embed_backward():
    """Backward of ``diag_embed``: the diagonal of the upstream gradient.

    The Triton kernel gathers the diagonal of ``grad_output`` into a
    contiguous gradient; the eager reference walks the ATen autograd chain
    (``DiagEmbedBackward0``), which returns a zero-copy view.
    """
    bench = DiagEmbedBackwardBenchmark(
        op_name="diag_embed_backward",
        input_fn=diag_embed_backward_input_fn,
        torch_op=_diag_embed_backward_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.diag_embed_backward)
    bench.run()
