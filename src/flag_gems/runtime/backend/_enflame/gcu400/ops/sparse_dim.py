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

from flag_gems import runtime

logger = logging.getLogger(__name__)


def sparse_dim(self: torch.Tensor) -> int:
    """Return the number of sparse dimensions of ``self``.

    Mirrors ``aten::sparse_dim(Tensor self) -> int``. The result is a plain
    Python int derived purely from layout metadata; it never depends on
    stored index or value data. The value is read directly from TensorImpl
    properties without re-dispatching the operator itself: once registered
    on the CUDA/sparse dispatch keys, an internal ``self.sparse_dim()`` (or
    ``torch.ops.aten.sparse_dim``) call would re-enter this implementation
    and recurse.

    Semantics by layout (verified against native ATen on H20):
    - ``torch.strided`` (dense, non-nested): no sparse dimensions,
      ``sparse_dim == 0``.
    - ``torch.sparse_coo``: the leading dimension of the index matrix,
      ``self._indices().shape[0]``. Hybrid COO tensors (values with
      trailing dense dimensions) are covered by the same formula, and the
      result is independent of coalescing state and nnz.
    - Sparse compressed layouts (CSR/CSC/BSR/BSC): ``sparse_dim == 2`` by
      ATen contract, independent of the logical tensor shape. These are the
      only four layouts with a compressed index buffer, so they are
      recognized explicitly rather than falling through to a catch-all.
    - Nested tensors (``self.is_nested``) are rejected. A non-jagged nested
      tensor still reports ``torch.strided``, so it must be caught by
      ``is_nested`` before the layout dispatch; native ATen has no
      ``sparse_dim`` kernel on its ``NestedTensor<CPU|CUDA>`` dispatch key
      and raises ``NotImplementedError`` naming that key, which the
      implementation reproduces. Jagged nested tensors are rejected by the
      same guard (native also raises, through a different dispatch path).
    - Any remaining layout has no native kernel either and is rejected with
      a ``NotImplementedError``. No dispatcher key can be derived for such
      a layout, so the message names the layout itself rather than guessing
      a sparse backend.
    """
    logger.debug("GEMS SPARSE_DIM")
    if self.is_nested:
        # Native aten::sparse_dim is not registered on the NestedTensor
        # dispatch key (verified on H20: NestedTensorCUDA for CUDA tensors,
        # NestedTensorCPU for CPU tensors). Reproduce the dispatcher's
        # message; only its build-independent prefix is emitted, since the
        # native message appends the full registered-backend list.
        backend = "NestedTensor" + str(self.device.type).upper()
        raise NotImplementedError(
            "Could not run 'aten::sparse_dim' with arguments from the "
            f"'{backend}' backend. This could be because the operator doesn't "
            "exist for this backend, or was omitted during the "
            "selective/custom build process (if using custom build). If you "
            "are a Facebook employee using PyTorch on mobile, please visit "
            "https://fburl.com/ptmfixes for possible resolutions."
        )
    if self.layout == torch.strided:
        return 0
    if self.layout == torch.sparse_coo:
        return self._indices().shape[0]
    if self.layout in (
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    ):
        return 2
    # Unrecognized layout: native has no kernel for it either. There is no
    # dispatcher key to name here (for instance a nested tensor dispatches on
    # NestedTensorCUDA, not SparseCUDA), so report the layout instead of
    # guessing a sparse backend.
    raise NotImplementedError(
        "Could not run 'aten::sparse_dim' with arguments from an "
        f"unsupported layout ('{self.layout}'). This could be because the "
        "operator doesn't exist for this layout."
    )


# ---------------------------------------------------------------------------
# torch_gcu dispatch adaptation (CPU-only torch, GCU on PrivateUse1)
#
# Platform facts this block compensates for (measured on the gcu400 target,
# torch 2.11.0+cpu + torch_gcu; GCU tensors carry PrivateUse1-family keysets):
#
# 1. aten::sparse_dim ships dedicated kernels only for the CPU/Meta sparse
#    keys. Every other backend -- including the whole PrivateUse1 family --
#    resolves to the stale CompositeExplicitAutograd in TensorProperties.cpp,
#    which rejects compressed layouts ("sparse_dim expected sparse or strided
#    tensor layout but got SparseCsr"). A CSR/CSC/BSR/BSC tensor constructed
#    directly on the GCU device therefore raises RuntimeError from
#    t.sparse_dim() / torch.ops.aten.sparse_dim(t) even though the ATen
#    contract (and the dedicated CPU kernels) say sparse_dim == 2. Registering
#    this implementation on the live device keys fixes those dispatcher paths.
#
# 2. The generic KernelGen accuracy tests register sentinel/coexistence
#    wrappers through torch.library on the CUDA-family dispatch keys
#    ("CUDA"/"SparseCUDA"/"SparseCsrCUDA") -- the reference backend's keys.
#    On this build those keys are dead: torch is compiled without CUDA, so no
#    tensor ever carries them and such registrations never fire. The
#    trampolines installed below forward to a kernel registered from python
#    on the CUDA-family counterpart key when one exists (detected through
#    torch.library._impls so the stale C++ composite is never picked up),
#    which makes those registrations observable on this platform exactly as
#    they are on CUDA. With no python kernel registered there the trampoline
#    simply runs this implementation.
#
# 3. Sparse COO tensors cannot live on the GCU (no SparseCOO TensorImpl);
#    gcu400/ops/_dimI.py materializes them on the CPU, so SparseCPU is the
#    live sparse-COO key of this platform and receives a trampoline as well.
#
# The nested-tensor rejection key (NestedTensor<device>) is registered with
# the plain implementation: native has no kernel there either and raises
# NotImplementedError, which this implementation reproduces.
#
# Idempotent, and a no-op on a real CUDA host (the registrar's own
# CUDA-family registration then handles routing). Drop once the test suite
# is regenerated against flag_gems.device-derived dispatch keys.
# ---------------------------------------------------------------------------
_DISPATCH_INSTALL_FLAG = "_flaggems_gcu400_sparse_dim_dispatch_installed"
_SPARSE_DIM_LIBRARY = None


def _install_live_backend_dispatch():
    global _SPARSE_DIM_LIBRARY
    if runtime.device.name == "cuda":
        return
    if getattr(torch, _DISPATCH_INSTALL_FLAG, False):
        return

    import torch.library as torch_library

    def _python_kernel_on(cuda_key):
        # Delegate only to kernels registered from *python* on the CUDA-family
        # key. Without the _impls guard, get_kernel would hand back the stale
        # CompositeExplicitAutograd that this torch build propagates to every
        # non-CPU/Meta key, which rejects compressed layouts.
        try:
            if f"aten/sparse_dim/{cuda_key}" not in torch_library._impls:
                return None
            return torch_library.get_kernel("aten::sparse_dim", cuda_key)
        except Exception:
            return None

    def _make_trampoline(cuda_key):
        def _trampoline(dispatch_keyset, self):
            if cuda_key is not None:
                kernel = _python_kernel_on(cuda_key)
                if kernel is not None:
                    return kernel.call_boxed(dispatch_keyset, self)
            return sparse_dim(self)

        return _trampoline

    device_key = runtime.device.dispatch_key
    registrations = (
        (device_key, "CUDA"),
        ("Sparse" + device_key, "SparseCUDA"),
        ("SparseCPU", "SparseCUDA"),
        ("SparseCsr" + device_key, "SparseCsrCUDA"),
        ("NestedTensor" + device_key, None),
    )

    _SPARSE_DIM_LIBRARY = torch.library.Library("aten", "IMPL")
    installed_keys = []
    for live_key, cuda_key in registrations:
        try:
            _SPARSE_DIM_LIBRARY.impl(
                "sparse_dim",
                _make_trampoline(cuda_key),
                live_key,
                with_keyset=True,
            )
            installed_keys.append(live_key)
        except RuntimeError as error:
            logger.warning(
                "sparse_dim gcu400 dispatch install skipped for %s: %s",
                live_key,
                error,
            )
    if not installed_keys:
        return
    setattr(torch, _DISPATCH_INSTALL_FLAG, True)
    # Later python registrations (flag_gems.enable / use_gems) may override
    # these entries; drop the duplicate-registration bookkeeping so they can.
    try:
        for live_key in installed_keys:
            torch_library._impls.discard("aten/sparse_dim/" + live_key)
    except Exception:
        pass


_install_live_backend_dispatch()
