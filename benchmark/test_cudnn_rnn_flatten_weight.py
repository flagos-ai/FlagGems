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

from . import base

# (input_size, hidden_size, num_layers)
_SHAPES = [
    (256, 256, 1),
    (512, 512, 2),
    (1024, 1024, 2),
]

# (module class, cudnn mode, proj_size, nonlinearity)
_RNN_MODES = [
    (torch.nn.RNN, 0, 0, "relu"),
    (torch.nn.RNN, 1, 0, "tanh"),
    (torch.nn.LSTM, 2, 0, None),
    (torch.nn.GRU, 3, 0, None),
    (torch.nn.LSTM, 2, 2, None),
]


class CudnnRnnFlattenWeightBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _SHAPES
        self.shape_desc = "input_size, hidden_size, num_layers"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            input_size, hidden_size, num_layers = shape
            for bidirectional in (False, True):
                for bias in (True, False):
                    for cls, mode, proj_size, nonlinearity in _RNN_MODES:
                        kwargs = {
                            "num_layers": num_layers,
                            "bidirectional": bidirectional,
                            "bias": bias,
                        }
                        if proj_size:
                            kwargs["proj_size"] = proj_size
                        if nonlinearity:
                            kwargs["nonlinearity"] = nonlinearity
                        module = cls(input_size, hidden_size, **kwargs).to(
                            dtype=cur_dtype, device=self.device
                        )
                        weights = [w.detach().clone() for w in module._flat_weights]
                        stride0 = (4 if bias else 2) + (1 if proj_size else 0)
                        yield (
                            weights,
                            stride0,
                            input_size,
                            mode,
                            hidden_size,
                            proj_size,
                            num_layers,
                            True,
                            bidirectional,
                        )


@pytest.mark.cudnn_rnn_flatten_weight
def test_cudnn_rnn_flatten_weight():
    bench = CudnnRnnFlattenWeightBenchmark(
        op_name="cudnn_rnn_flatten_weight",
        torch_op=getattr(torch.ops, "aten")._cudnn_rnn_flatten_weight,
        # bfloat16 weights are rejected by the cuDNN reference implementation.
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


class CudnnRnnFlattenWeightOutBenchmark(CudnnRnnFlattenWeightBenchmark):
    """Same inputs, but the result is written into a preallocated ``out``."""

    def get_input_iter(self, cur_dtype):
        for args in super().get_input_iter(cur_dtype):
            weights = args[0]
            # The flat buffer size is fixed by the layout, so ask the reference
            # implementation once (outside the timed region) for the length.
            size = torch.ops.aten._cudnn_rnn_flatten_weight(
                [w.clone() for w in weights], *args[1:]
            ).numel()
            out = torch.empty(size, dtype=weights[0].dtype, device=self.device)
            yield args + ({"out": out},)


@pytest.mark.cudnn_rnn_flatten_weight_out
def test_cudnn_rnn_flatten_weight_out():
    bench = CudnnRnnFlattenWeightOutBenchmark(
        op_name="cudnn_rnn_flatten_weight_out",
        torch_op=getattr(torch.ops, "aten")._cudnn_rnn_flatten_weight.out,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
