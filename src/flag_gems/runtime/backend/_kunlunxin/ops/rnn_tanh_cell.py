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

"""Kunlunxin rnn_tanh_cell override: empty-reduction (input_size == 0) fix.

The generic ``_rnn_tanh_cell_kernel`` trips a triton-xpu backend defect when
the input-reduction dimension is empty: the zero-trip ``k`` loop yields
wrong and non-deterministic outputs on this platform (forcing a single
masked iteration instead crashes the kernel with KL3 error 719). Host-side
padding to one zero column keeps the math identical -- the dot contributes
exactly zero and the ``b_ih`` bias is retained, matching the ATen eager
semantics the accuracy tests reference -- and lets the kernel take the
well-tested non-empty path.
"""

import torch

from flag_gems.ops.rnn_tanh_cell import rnn_tanh_cell as _cell

__all__ = ["rnn_tanh_cell"]


def rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    """Single tanh RNN cell step (kunlunxin)."""
    if input.shape[1] == 0:
        input = torch.zeros((input.shape[0], 1), dtype=input.dtype, device=input.device)
        w_ih = torch.zeros((w_ih.shape[0], 1), dtype=w_ih.dtype, device=w_ih.device)
    return _cell(input, hx, w_ih, w_hh, b_ih, b_hh)
