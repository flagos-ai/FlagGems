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

import flag_gems

from . import base, consts, utils
from .conftest import Config


class BeamSearchScoreBenchmark(base.Benchmark):
    """
    Benchmark for Beam Search Score operation.
    """

    # Use smaller shapes suitable for beam search scenarios
    DEFAULT_SHAPES = [(16, 512), (32, 1024), (64, 2048), (128, 4096), (256, 8192)]
    DEFAULT_SHAPE_DESC = "batch_size, vocab_size"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            # log_probs: [batch, vocab], beam_scores: [batch]
            batch_size = shape[0]
            vocab_size = shape[1] if len(shape) > 1 else shape[0]
            log_probs = utils.generate_tensor_input(
                (batch_size, vocab_size), cur_dtype, self.device
            )
            beam_scores = utils.generate_tensor_input(
                (batch_size,), cur_dtype, self.device
            )
            yield log_probs, beam_scores

    def get_tflops(self, op, *args, **kwargs):
        shape1 = list(args[0].shape)  # log_probs shape
        shape2 = list(args[1].shape)  # beam_scores shape
        return torch.tensor(shape1).prod().item() + torch.tensor(shape2).prod().item()

    def init_user_config(self):
        # Override to skip reading from YAML and use DEFAULT_SHAPES directly
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC


@pytest.mark.beam_search_score
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_beam_search_score(dtype):
    if flag_gems.vendor_name == "metax":
        pytest.skip("Metax backend CI validates correctness; skip backend benchmark.")

    # Reference implementation: PyTorch broadcasting addition
    def torch_op(log_probs, beam_scores):
        return log_probs + beam_scores.unsqueeze(-1)

    bench = BeamSearchScoreBenchmark(
        op_name="beam_search_score",
        torch_op=torch_op,
        dtypes=[dtype],
    )
    bench.gems_op = flag_gems.beam_search_score
    bench.run()


@pytest.mark.beam_search_score_
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_beam_search_score_(dtype):
    if flag_gems.vendor_name == "metax":
        pytest.skip("Metax backend CI validates correctness; skip backend benchmark.")

    # Reference implementation: PyTorch broadcasting addition
    def torch_op(log_probs, beam_scores):
        return log_probs + beam_scores.unsqueeze(-1)

    bench = BeamSearchScoreBenchmark(
        op_name="beam_search_score_",
        torch_op=torch_op,
        dtypes=[dtype],
        is_inplace=True,
    )
    bench.gems_op = flag_gems.beam_search_score_
    bench.run()
