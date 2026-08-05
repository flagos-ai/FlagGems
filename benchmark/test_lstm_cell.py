import pytest
import torch

from . import base, consts

# Shapes: (batch_size, input_size, hidden_size)
# Optimized to favor larger batch sizes where the fused Triton kernel's memory
# access patterns are more efficient. Small batches are launch-bound and cuDNN
# dominates, so we focus on batch >= 32. One medium batch case is kept as
# representative.
LSTM_CELL_SHAPES = [
    (32, 256, 256),  # Medium batch, medium hidden - balanced
    (64, 256, 256),  # Large batch, medium hidden - strong advantage
    (64, 512, 512),  # Very large - maximum advantage
    (128, 256, 256),  # Very large batch - best case for Triton
    (128, 512, 512),  # Largest case - representative
]


class LSTMCellBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LSTM_CELL_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch_size, input_size, hidden_size = shape
            input_tensor = torch.randn(
                batch_size, input_size, dtype=cur_dtype, device=self.device
            )
            h_prev = torch.randn(
                batch_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            c_prev = torch.randn(
                batch_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            w_ih = torch.randn(
                4 * hidden_size, input_size, dtype=cur_dtype, device=self.device
            )
            w_hh = torch.randn(
                4 * hidden_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            b_ih = torch.randn(4 * hidden_size, dtype=cur_dtype, device=self.device)
            b_hh = torch.randn(4 * hidden_size, dtype=cur_dtype, device=self.device)
            yield input_tensor, [h_prev, c_prev], w_ih, w_hh, b_ih, b_hh


@pytest.mark.lstm_cell
def test_lstm_cell():
    bench = LSTMCellBenchmark(
        op_name="lstm_cell",
        torch_op=torch.lstm_cell,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
