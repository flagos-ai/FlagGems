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

logger = logging.getLogger(__name__)


def _functional_sym_constrain_range_for_size(*args, **kwargs):
    r"""Pass through the dep_token of a symbolic-shape marker op.

    ``aten::_functional_sym_constrain_range_for_size(Scalar size, int? min,
    int? max, Tensor dep_token) -> Tensor`` only records a size range hint for
    symbolic shape tracing. At eager runtime nothing is produced or mutated, so
    no device kernel is launched (same contract as metadata-only ops such as
    ``aten::alias``) and the dep_token is returned as-is.
    """
    logger.debug("GEMS_ENFLAME FUNCTIONAL_SYM_CONSTRAIN_RANGE_FOR_SIZE")
    tensor_arg = None
    for a in args:
        if isinstance(a, torch.Tensor):
            tensor_arg = a
            break
    if tensor_arg is None:
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                tensor_arg = v
                break

    if tensor_arg is not None:
        return tensor_arg

    return args[0] if len(args) > 0 else None
