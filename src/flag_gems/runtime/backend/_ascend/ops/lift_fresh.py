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

import torch

from flag_gems.ops.lift_fresh import lift_fresh as _generic_lift_fresh
from flag_gems.ops.lift_fresh_copy import (
    lift_fresh_copy as _generic_lift_fresh_copy,
)
from flag_gems.ops.lift_fresh_copy import (
    lift_fresh_copy_out as _generic_lift_fresh_copy_out,
)


def _input_tensor(args, kwargs):
    return next(
        (value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)),
        None,
    )


def lift_fresh(*args, **kwargs):
    x = _input_tensor(args, kwargs)
    if x is not None and x.numel() == 0:
        return x.clone()
    return _generic_lift_fresh(*args, **kwargs)


def lift_fresh_copy(*args, **kwargs):
    x = _input_tensor(args, kwargs)
    if x is not None and x.numel() == 0:
        return x.clone()
    return _generic_lift_fresh_copy(*args, **kwargs)


def lift_fresh_copy_out(x: torch.Tensor, out: torch.Tensor = None):
    if x.numel() == 0:
        if out is None:
            return x.clone()
        if out.dtype != x.dtype or out.device != x.device:
            raise ValueError("Output tensor must match input dtype and device")
        out.resize_(x.shape)
        return out
    return _generic_lift_fresh_copy_out(x, out)
