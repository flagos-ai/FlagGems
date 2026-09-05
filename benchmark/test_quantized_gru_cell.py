# Copyright 2026 FlagOS Contributors.
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

from typing import Generator

import pytest
import torch

from flag_gems.ops.quantized_gru_cell import (
    quantized_gru_cell as gems_quantized_gru_cell,
)

from . import base, consts

# (batch, input_size, hidden_size) shapes for the quantized GRU cell.
GRU_SHAPES = [
    (1, 8, 4),
    (8, 32, 16),
    (32, 64, 32),
    (64, 128, 64),
    (128, 256, 128),
]


def _torch_quantized_gru_cell_ref(
    input,
    hx,
    w_ih,
    w_hh,
    b_ih,
    b_hh,
    packed_ih,
    packed_hh,
    col_offsets_ih,
    col_offsets_hh,
    scale_ih,
    scale_hh,
    zero_point_ih,
    zero_point_hh,
):
    """CUDA-runnable mathematical reference for ``aten::quantized_gru_cell``.

    The aten op is CPU-only (it dispatches to the FBGEMM packed kernel), so
    for a GPU latency baseline we run the mathematically-equivalent plain
    ``torch.gru_cell`` on dequantized weights ``scale * (w_q - zero_point)``.
    This is exactly what the fused Triton kernel computes.
    """
    w_ih_deq = scale_ih * (w_ih.to(torch.float32) - zero_point_ih)
    w_hh_deq = scale_hh * (w_hh.to(torch.float32) - zero_point_hh)
    return torch.gru_cell(
        input.to(torch.float32),
        hx.to(torch.float32),
        w_ih_deq,
        w_hh_deq,
        b_ih.to(torch.float32),
        b_hh.to(torch.float32),
    ).to(input.dtype)


def quantized_gru_cell_input_fn(shape, dtype, device):
    batch_size, input_size, hidden_size = shape
    inp = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    b_ih = torch.randn(3 * hidden_size, dtype=dtype, device=device)
    b_hh = torch.randn(3 * hidden_size, dtype=dtype, device=device)

    # Quantize weights on CPU (fbgemm helpers are CPU-only) then move the
    # int8 weight + col_offsets to the benchmark device. The packed weights
    # are not needed by either kernel path but are passed to keep the
    # signature identical to the aten op.
    w_ih_float = torch.randn(3 * hidden_size, input_size, dtype=torch.float32)
    w_hh_float = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32)
    w_ih_q, col_ih, scale_ih, zp_ih = torch.fbgemm_linear_quantize_weight(w_ih_float)
    w_hh_q, col_hh, scale_hh, zp_hh = torch.fbgemm_linear_quantize_weight(w_hh_float)
    w_ih = w_ih_q.to(device)
    w_hh = w_hh_q.to(device)
    col_offsets_ih = col_ih.to(device)
    col_offsets_hh = col_hh.to(device)
    packed_ih = torch.empty(0, device=device)
    packed_hh = torch.empty(0, device=device)
    yield (
        inp,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_offsets_ih,
        col_offsets_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )


class QuantizedGruCellBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = GRU_SHAPES

    def set_more_shapes(self):
        # The base GenericBenchmark appends huge 1D/2D/3D shapes that don't
        # fit the (batch, input_size, hidden_size) layout; keep our own.
        return []

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.quantized_gru_cell
def test_quantized_gru_cell():
    bench = QuantizedGruCellBenchmark(
        input_fn=quantized_gru_cell_input_fn,
        op_name="quantized_gru_cell",
        torch_op=_torch_quantized_gru_cell_ref,
        gems_op=gems_quantized_gru_cell,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
