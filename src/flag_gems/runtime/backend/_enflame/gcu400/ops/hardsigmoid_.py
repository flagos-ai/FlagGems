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


def hardsigmoid_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME HARDSIGMOID_")
    from .hardsigmoid import _flat_path_ok, _launch_hardsigmoid

    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))

    if x is None:
        raise ValueError("hardsigmoid_: expected a Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("hardsigmoid_: expected a Tensor")
    if not x.is_floating_point():
        raise TypeError("hardsigmoid_: expected a floating point tensor")

    if not _flat_path_ok(x):
        from flag_gems.ops.hardsigmoid_ import hardsigmoid_ as _public

        return _public(x)

    _launch_hardsigmoid(x, x)
    return x
