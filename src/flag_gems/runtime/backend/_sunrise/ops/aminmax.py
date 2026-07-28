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


def _aminmax_cpu_reference(op_name, inp, *args, out=None, **kwargs):
    cpu_inp = inp.cpu()
    cpu_args = tuple(
        arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args
    )
    cpu_kwargs = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
    }
    cpu_result = getattr(torch, op_name)(cpu_inp, *cpu_args, **cpu_kwargs)

    if out is None:
        if isinstance(cpu_result, tuple):
            return tuple(item.to(device=inp.device) for item in cpu_result)
        return cpu_result.to(device=inp.device)

    if isinstance(out, tuple):
        for cpu_item, out_item in zip(cpu_result, out):
            out_item.copy_(cpu_item.to(device=out_item.device))
        return out

    out.copy_(cpu_result.to(device=out.device))
    return out


def amin(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_SUNRISE AMIN_CPU_REFERENCE")
    return _aminmax_cpu_reference("amin", inp, dim=dim, keepdim=keepdim, out=out)


def amin_out(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_SUNRISE AMIN_OUT")
    if out is None:
        raise ValueError("amin_out expects an out tensor")
    return amin(inp, dim=dim, keepdim=keepdim, out=out)


def amax(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_SUNRISE AMAX")
    return _aminmax_cpu_reference("amax", inp, dim=dim, keepdim=keepdim, out=out)


def amax_out(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_SUNRISE AMAX_OUT")
    if out is None:
        raise ValueError("amax_out expects an out tensor")
    return amax(inp, dim=dim, keepdim=keepdim, out=out)


def aminmax(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_SUNRISE AMINMAX")
    return _aminmax_cpu_reference("aminmax", inp, dim=dim, keepdim=keepdim, out=out)


def aminmax_out(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_SUNRISE AMINMAX_OUT")
    if out is None:
        raise ValueError("aminmax_out expects an out tuple")
    return aminmax(inp, dim=dim, keepdim=keepdim, out=out)
