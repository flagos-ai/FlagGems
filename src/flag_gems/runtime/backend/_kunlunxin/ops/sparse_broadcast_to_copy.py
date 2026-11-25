"""Kunlunxin override of ``aten::_sparse_broadcast_to_copy``.

The copy variant shares the broadcasting semantics of
``aten::_sparse_broadcast_to`` (materialised by the backend override) and adds
the ``out=`` form.  The generic module imports the generic
``sparse_broadcast_to`` implementation at import time, so it would keep using
the kernel that fails MLIR legalization on this backend; this override routes
through the backend implementation instead.
"""

import logging

from flag_gems.runtime.backend._kunlunxin.ops.sparse_broadcast_to import (
    sparse_broadcast_to,
)

logger = logging.getLogger(__name__)


def sparse_broadcast_to_copy(self, size, out=None):
    logger.debug("GEMS SPARSE BROADCAST TO COPY (kunlunxin)")
    result = sparse_broadcast_to(self, size)
    if out is not None:
        out.copy_(result)
        return out
    return result
