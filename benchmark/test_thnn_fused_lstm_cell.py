import pytest
import torch

from . import base, consts

# Shapes for LSTM cell: (batch, hidden_size) where gates = (batch, 4*hidden_size)
LSTM_SHAPES = [
    (1, 4),
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
]


class LSTMCellBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LSTM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch_size, hidden_size = shape
            input_gates = torch.randn(
                batch_size, 4 * hidden_size, dtype=cur_dtype, device=self.device
            )
            hidden_gates = torch.randn(
                batch_size, 4 * hidden_size, dtype=cur_dtype, device=self.device
            )
            cx = torch.randn(
                batch_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            yield input_gates, hidden_gates, cx


@pytest.mark.thnn_fused_lstm_cell
def test_thnn_fused_lstm_cell():
    bench = LSTMCellBenchmark(
        op_name="thnn_fused_lstm_cell",
        torch_op=torch.ops.aten._thnn_fused_lstm_cell,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
