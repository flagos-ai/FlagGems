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

"""Hygon specialization for scatter_reduce product reduction."""

import logging

from flag_gems.ops.scatter_reduce import (
    _scatter_reduce_high_rank,
    _scatter_reduce_prod_scan,
)
from flag_gems.ops.scatter_reduce import scatter_reduce as _generic_scatter_reduce

logger = logging.getLogger(__name__)

_MAX_SCAN_OUTPUTS = 65535
# The deterministic product-scan kernel launches one Triton program per output
# element. HCU limits each launch-grid axis to 65,535 programs, so larger
# outputs use the source-driven lock kernel and spill excess programs onto a
# second axis. The threshold only selects the path and caps grid-x; it does not
# change scatter_reduce semantics.


def _use_prod_scan(inp, reduce):
    return reduce == "prod" and inp.numel() <= _MAX_SCAN_OUTPUTS


def _scatter_reduce_prod(inp, dim, index, src, include_self):
    if inp.ndim > 5:
        return _scatter_reduce_high_rank(
            inp,
            dim,
            index,
            src,
            "prod",
            include_self,
            use_prod_scan=True,
        )
    return _scatter_reduce_prod_scan(inp, dim, index, src, include_self)


def _scatter_reduce(inp, dim, index, src, reduce, include_self):
    if _use_prod_scan(inp, reduce):
        return _scatter_reduce_prod(inp, dim, index, src, include_self)
    return _generic_scatter_reduce(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self=include_self,
        _use_product_lock=reduce == "prod",
        _product_grid_limit=_MAX_SCAN_OUTPUTS,
    )


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_HYGON SCATTER_REDUCE_TWO")
    return _scatter_reduce(inp, dim, index, src, reduce, include_self)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_HYGON SCATTER_REDUCE_TWO_")
    result = _scatter_reduce(inp, dim, index, src, reduce, include_self)
    inp.copy_(result)
    return inp


def scatter_reduce_out(
    inp,
    dim,
    index,
    src,
    reduce,
    *,
    include_self=True,
    out=None,
):
    logger.debug("GEMS_HYGON SCATTER_REDUCE_TWO_OUT")
    if out is not None and out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    result = _scatter_reduce(inp, dim, index, src, reduce, include_self)
    if out is not None:
        out.copy_(result)
        return out
    return result
