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

from .base import Benchmark


class CtcLossBackwardBenchmark(Benchmark):
    """
    Benchmark for _ctc_loss_backward operator.

    Since backward requires forward outputs, we run forward once during setup
    and reuse neg_log_likelihood and log_alpha for all backward timing runs.
    """

    # CTC loss backward only supports float32 and float64
    DEFAULT_DTYPES = [torch.float32, torch.float64]

    def set_shapes(self, shape_file_path=None):
        # (T, N, S, C) - timesteps, batch, target_len, num_classes
        self.shapes = [
            (64, 4, 32, 32),
            (256, 16, 64, 64),
            (512, 32, 64, 64),
            (1024, 32, 128, 128),
        ]
        self.shape_desc = "T, N, S, C"

    def get_input_iter(self, cur_dtype):
        for cur_shape in self.shapes:
            T, N, S, C = cur_shape

            # Forward inputs
            log_probs = torch.randn(T, N, C, dtype=cur_dtype, device=self.device)
            targets = torch.randint(1, C, (N, S), dtype=torch.long, device=self.device)
            input_lengths = torch.full((N,), T, dtype=torch.long, device=self.device)
            target_lengths = torch.randint(
                S // 2, S + 1, (N,), dtype=torch.long, device=self.device
            )

            # Run forward to get neg_log_likelihood and log_alpha (needed for backward)
            with flag_gems.use_gems():
                neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
                    log_probs, targets, input_lengths, target_lengths
                )

            # Upstream gradient
            grad_output = torch.randn_like(neg_log_likelihood)

            # Yield backward inputs with blank parameter
            yield grad_output, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, {
                "blank": 0
            }

    def torch_forward(
        self,
        grad_output,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
    ):
        return torch.ops.aten._ctc_loss_backward(
            grad_output,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            neg_log_likelihood,
            log_alpha,
        )

    def gems_forward(
        self,
        grad_output,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
    ):
        with flag_gems.use_gems():
            return torch.ops.aten._ctc_loss_backward(
                grad_output,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                neg_log_likelihood,
                log_alpha,
            )


@pytest.mark.ctc_loss_backward_internal
def test_perf__ctc_loss_backward():
    bench = CtcLossBackwardBenchmark(
        op_name="_ctc_loss_backward",
        torch_op=torch.ops.aten._ctc_loss_backward,
        arg_func=None,
    )
    bench.run()
