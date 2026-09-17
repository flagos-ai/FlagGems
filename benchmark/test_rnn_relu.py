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

from typing import Generator

import pytest
import torch

from . import base, consts


def rnn_relu_input_fn(shape, dtype, device):
    seq_len, batch_size, input_size = shape
    hidden_size = input_size
    inp = torch.randn(seq_len, batch_size, input_size, dtype=dtype, device=device)
    hx = torch.randn(1, batch_size, hidden_size, dtype=dtype, device=device)
    rnn = torch.nn.RNN(input_size, hidden_size, 1, nonlinearity="relu")
    rnn = rnn.to(dtype=dtype, device=device)
    params = tuple(rnn._flat_weights)
    yield inp, {
        "hx": hx,
        "params": params,
        "has_biases": True,
        "num_layers": 1,
        "dropout": 0.0,
        "train": False,
        "bidirectional": False,
        "batch_first": False,
    }


class RnnReluBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes = [
            (16, 4, 32),
            (32, 8, 64),
            (64, 16, 128),
        ]
        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.rnn_relu
def test_rnn_relu():
    bench = RnnReluBenchmark(
        input_fn=rnn_relu_input_fn,
        op_name="rnn_relu",
        torch_op=torch.rnn_relu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
