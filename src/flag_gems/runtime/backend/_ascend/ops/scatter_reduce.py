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

"""Ascend specialization for scatter_reduce product reduction."""

import logging

from flag_gems.ops.scatter_reduce import _scatter_reduce_prod_scan
from flag_gems.ops.scatter_reduce import scatter_reduce as _generic_scatter_reduce
from flag_gems.ops.scatter_reduce import scatter_reduce_ as _generic_scatter_reduce_
from flag_gems.ops.scatter_reduce import (
    scatter_reduce_out as _generic_scatter_reduce_out,
)

logger = logging.getLogger(__name__)


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    if reduce != "prod":
        return _generic_scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
        )
    logger.debug("GEMS_ASCEND SCATTER_REDUCE_TWO")
    assert inp.ndim <= 5, f"scatter_reduce supports up to 5D tensors, got {inp.ndim}D"
    return _scatter_reduce_prod_scan(
        inp,
        dim,
        index,
        src,
        include_self,
        materialize_product=True,
    )


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    if reduce != "prod":
        return _generic_scatter_reduce_(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
        )
    logger.debug("GEMS_ASCEND SCATTER_REDUCE_TWO_")
    result = _scatter_reduce_prod_scan(
        inp,
        dim,
        index,
        src,
        include_self,
        materialize_product=True,
    )
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
    if reduce != "prod":
        return _generic_scatter_reduce_out(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
            out=out,
        )
    logger.debug("GEMS_ASCEND SCATTER_REDUCE_TWO_OUT")
    if out is not None and out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    result = _scatter_reduce_prod_scan(
        inp,
        dim,
        index,
        src,
        include_self,
        materialize_product=True,
    )
    if out is not None:
        out.copy_(result)
        return out
    return result
