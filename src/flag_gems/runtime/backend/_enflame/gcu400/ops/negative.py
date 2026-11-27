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

import logging

import torch

from .neg import _run_neg
from .neg import neg as negative
from .neg import neg_ as negative_

logger = logging.getLogger(__name__)

__all__ = ["negative", "negative_", "negative_out"]


def negative_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME NEGATIVE_OUT")
    if (
        x.dtype == out.dtype
        and x.is_contiguous()
        and out.is_contiguous()
        and out.numel() == x.numel()
    ):
        _run_neg(x, out)
        return out

    # dtype-cast / non-contiguous staging fallback; keeps the pre-change
    # semantics (negate computed in out.dtype, out layout preserved).
    x_contig = x.to(out.dtype).contiguous()
    if out.is_contiguous() and out.numel() == x_contig.numel():
        _run_neg(x_contig, out)
    else:
        out_contig = torch.empty_like(x_contig)
        _run_neg(x_contig, out_contig)
        out.copy_(out_contig)
    return out
