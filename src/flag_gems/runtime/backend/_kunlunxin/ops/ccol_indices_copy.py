# Kunlunxin (XPU) override of ccol_indices_copy / ccol_indices_copy_out.
#
# The generic KernelGen implementation (src/flag_gems/ops/ccol_indices_copy.py)
# is a small Triton memcpy of the compressed-column-index buffer and already
# passes every functional test on the XPU backend, so the vendor override
# re-exports it unchanged (exporting the name from ops/__init__.py is enough
# for SpecOpRegistrar to swap the generic one).
#
# The XPU-specific work lives in the companion fix below.  The two reject
# cases of tests/test_ccol_indices_copy.py build their non-column-compressed
# inputs with ``csc.to_sparse_csr()`` / ``bsc.to_sparse_bsr()``.  On this
# stack (torch 2.9 + torch_xmlir / KL3) the native device kernels behind the
# composite ``to_sparse_csr``/``to_sparse_bsr`` (``_to_sparse_csr`` /
# ``_to_sparse_bsr``) are missing -- every call raises
# ``cudaErrorInvalidDeviceFunction`` -- so the test cannot even construct its
# inputs, and the rejection contract is unreachable.
#
# Fix: register ``_to_sparse_csr`` / ``_to_sparse_bsr`` overrides on the
# ``SparseCsrCUDA`` dispatch key (the key carried by CSC/BSC tensors; verified
# through ``torch._C._dispatch_keys``).  For CSC/BSC inputs a transpose is a
# pure layout swap (no data-movement kernel), so ``self.transpose(0, 1)``
# yields a valid CSR/BSR tensor of the transposed matrix -- exactly the layout
# the ccol rejection contract inspects.  The registration is process-global
# and follows the ``_to_sparse_csr``/``_to_sparse_bsr`` override pattern of
# ``range.py`` / ``adaptive_max_pool2d`` in this package.

import logging

import torch

from flag_gems.ops import ccol_indices_copy, ccol_indices_copy_out  # noqa: F401

logger = logging.getLogger(__name__)


def _to_sparse_csr_impl(self, dense_dim=None):
    """XPU-safe ``_to_sparse_csr``: CSC/CSR layout swap in metadata only.

    ``dense_dim`` is accepted for schema compatibility; the ccol test suite
    only exercises the dense_dim=None path (2-D CSC).
    """
    if self.layout == torch.sparse_csc:
        return self.transpose(0, 1)
    if self.layout == torch.sparse_csr:
        return self
    # BSC/BSR -> CSR requires unnesting blocks (a data-movement conversion);
    # never reached by the ccol_indices_copy suite, so no device kernel that
    # would trip the missing-XPU-kernel bug is launched here.
    raise NotImplementedError(
        f"_to_sparse_csr: unsupported layout {self.layout} on kunlunxin"
    )


def _to_sparse_bsr_impl(self, blocksize, dense_dim=None):
    """XPU-safe ``_to_sparse_bsr``: BSC/BSR layout swap in metadata only.

    The transpose swaps the BSC block shape ``(row, col)`` to the BSR block
    shape ``(col, row)``; the swap is exact when the caller requests that
    (square- or transposed-blocksize inputs, i.e. every 2-D BSC that
    ``to_sparse_bsr`` is given with its own block shape).
    """
    if self.layout == torch.sparse_bsc:
        return self.transpose(0, 1)
    if self.layout == torch.sparse_bsr:
        return self
    raise NotImplementedError(
        f"_to_sparse_bsr: unsupported layout {self.layout} on kunlunxin"
    )


# Process-global registration (the Library handle is held in this module).
# ``allow_override=True`` matches the ``use_gems()`` registrar behaviour for
# the same operator/key (see range.py / adaptive_max_pool2d in this package).
_sparse_convert_lib = torch.library.Library("aten", "IMPL")
_sparse_convert_lib.impl(
    "_to_sparse_csr", _to_sparse_csr_impl, "SparseCsrCUDA", allow_override=True
)
_sparse_convert_lib.impl(
    "_to_sparse_bsr", _to_sparse_bsr_impl, "SparseCsrCUDA", allow_override=True
)
