# Copyright 2026 The FlagOS Contributors
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

"""Single-step ReLU RNN cell, kunlunxin wrapper.

The shared kernel reduces the input projection with ``tl.dot`` over
``input_size``.  When ``input_size == 0`` that loop makes zero trips, which this
backend compiles to an undefined accumulator (the whole output tile is garbage).
A one-column zero input and a matching zero weight column are exactly
equivalent -- the projection contributes only its bias -- but the reduction
then has a single trip, so the shared kernel is used unchanged for every other
shape.
"""

import logging

import torch

from flag_gems.ops.rnn_relu_cell import rnn_relu_cell as _rnn_relu_cell

logger = logging.getLogger(__name__)


def rnn_relu_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    logger.debug("GEMS_KUNLUNXIN RNN_RELU_CELL")
    if input.dim() >= 2 and input.shape[1] == 0:
        input = torch.zeros((input.shape[0], 1), dtype=input.dtype, device=input.device)
        w_ih = torch.zeros((w_ih.shape[0], 1), dtype=w_ih.dtype, device=w_ih.device)
    return _rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh)
