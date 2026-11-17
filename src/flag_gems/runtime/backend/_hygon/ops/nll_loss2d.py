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

"""Hygon ATen entry point for spatial NLL loss."""

import torch

from .nll_loss_nd import nll_loss_forward


def nll_loss2d(self, target, weight=None, reduction=1, ignore_index=-100):
    """Build the Hygon NLL autograd node and return its loss output."""
    assert self.ndim == 4, "nll_loss2d: expected 4D input (N, C, H, W)"
    assert self.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"nll_loss2d: unsupported dtype {self.dtype}"
    if target.shape != self.shape[:1] + self.shape[2:]:
        raise ValueError("nll_loss2d: target shape must match (N, H, W)")
    if weight is not None and (weight.ndim != 1 or weight.numel() != self.shape[1]):
        raise ValueError("nll_loss2d: weight must have one entry per class")

    loss, _ = nll_loss_forward(self, target, weight, int(reduction), int(ignore_index))
    return loss
