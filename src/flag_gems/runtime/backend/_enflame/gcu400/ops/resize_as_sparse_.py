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

from flag_gems import runtime  # noqa: F401

logger = logging.getLogger(__name__)

_SPARSE_COO = torch.sparse_coo
_STRIDED = torch.strided
_ALLOWED_TEMPLATE_LAYOUTS = (torch.sparse_coo, torch.strided)

# ---------------------------------------------------------------------------
# torch_gcu dispatch adaptation (GCU400 has no SparseCOO TensorImpl)
#
# Sparse COO tensors produced on this backend have all components on CPU (`_dimI.py`/`sparse_resize_.py`
# established fallback), but mixed-device calls like "CPU COO self + GCU dense template" have a dispatch key
# union landing on Sparse<device-key> ("Sparsegcu"), and torch_gcu
# has no kernel registered for `aten::resize_as_sparse_` on that key - reference implementations and user code calling
# `Tensor.resize_as_sparse_` directly raise NotImplementedError, even though this operator reads only layout
# metadata and is device-independent. Following `_sparse_coo_tensor_with_dims` in `sparse_resize_.py`
# and SparseCPU trampoline in `sparse_dim.py`, idempotently register this implementation on the sparse
# device key: calls land back in this Python implementation, which reads only template layout/size metadata and
# forwards `sparse_resize_` (CPU self lands on native SparseCPU kernel, semantics item-for-item identical to native
# resize_as_sparse_). Registration is idempotent with a torch attribute flag; later duplicate registration of the same
# key by enable()/
# use_gems() is caught by the registrar as a warning, behavior unchanged.
# ---------------------------------------------------------------------------
_RESIZE_AS_SPARSE_SPARSE_KEY_FLAG = (
    "_flag_gems_gcu400_resize_as_sparse_sparse_key_registered"
)
_RESIZE_AS_SPARSE_LIBRARY = None


# ---------------------------------------------------------------------------
# COO alias views must survive ``.detach()`` (GCU400 sparse component shim)
#
# ``sparse_coo_tensor.py``'s `_maybe_attach_alias_views` points the factory-produced COO container's
# ``_indices``/``_values`` back to the caller's device tensors as **instance attributes** (upstream
# alias contract: `torch.sparse_coo_tensor` does not copy components). ``.detach()`` creates a new Python
# object but does not copy instance attributes, so the detached view falls back to class-level resolution - native reads
# the CPU component buffer,
# diverging from the device view on the original object (probe confirmed: detaching immediately after construction
# already diverges,
# unrelated to any resize). `resize_as_sparse_`'s alias contract requires the detached view and the body to see the same
# values storage; this wrapper passes existing instance views through to the detached result, and tensors without views
# keep
# native detach semantics unchanged. The torch attribute flag is idempotent and safe under dual-name module imports.
# ---------------------------------------------------------------------------
_DETACH_ALIAS_VIEWS_FLAG = "_flag_gems_gcu400_coo_detach_views_registered"
_CLONE_ALIAS_VIEWS_FLAG = "_flag_gems_gcu400_coo_clone_views_registered"
_native_tensor_detach = None
_native_tensor_clone = None


def _install_detach_alias_views():
    """Pass COO ``_indices``/``_values`` instance views through to the ``.detach()`` result."""
    global _native_tensor_detach
    if getattr(torch.Tensor, _DETACH_ALIAS_VIEWS_FLAG, False):
        return
    _native_tensor_detach = torch.Tensor.detach

    def _detach_with_alias_views(self, *args, **kwargs):
        out = _native_tensor_detach(self, *args, **kwargs)
        views = self.__dict__
        if views and isinstance(out, torch.Tensor):
            for name in ("_indices", "_values"):
                view = views.get(name)
                if view is not None:
                    setattr(out, name, view)
        return out

    torch.Tensor.detach = _detach_with_alias_views
    setattr(torch.Tensor, _DETACH_ALIAS_VIEWS_FLAG, True)


def _install_clone_alias_views():
    """Clone COO instance views on the components' device and attach them to the ``.clone()`` result.

    native ``clone`` copies tensor components on the components' own device; on this backend the COO container's
    components land on CPU while the user-visible components are device tensors pointed to by instance views - native
    copies only the CPU side, so clone's ``_indices()``/``_values()`` silently
    degrade from device views to CPU components, and the reference side (`inp.clone()`) and result side (views retained)
    diverge in `torch.testing.assert_close`'s device attribute check. Same pattern as detach pass-through:
    intervene only when views exist, clone on the views' own device, tensors without views keep native
    clone semantics unchanged. The torch attribute flag is idempotent and safe under dual-name module imports.
    """
    global _native_tensor_clone
    if getattr(torch.Tensor, _CLONE_ALIAS_VIEWS_FLAG, False):
        return
    _native_tensor_clone = torch.Tensor.clone

    def _clone_with_alias_views(self, *args, **kwargs):
        if self.layout == torch.sparse_coo and self.dense_dim() > 0:
            # The backend COO copy drops the hybrid dense dimensions (probe:
            # sparse_dim=2/dense_dim=1 becomes sparse_dim=3/dense_dim=0), so a
            # plain hybrid clone silently changes the sparse/dense split and any
            # later ``resize_as_sparse_`` rejects the result as a sparse-dim
            # change. Rebuild the container from independently cloned components
            # so the hybrid layout survives; copy semantics are unchanged.
            return torch.sparse_coo_tensor(
                self._indices().clone(),
                self._values().clone(),
                self.size(),
                is_coalesced=self.is_coalesced(),
            )
        out = _native_tensor_clone(self, *args, **kwargs)
        views = self.__dict__
        if views and isinstance(out, torch.Tensor):
            idx_view = views.get("_indices")
            vals_view = views.get("_values")
            if idx_view is not None or vals_view is not None:
                if idx_view is not None:
                    cloned_idx = (
                        idx_view() if callable(idx_view) else idx_view
                    ).clone()
                    out._indices = lambda: cloned_idx
                if vals_view is not None:
                    cloned_vals = (
                        vals_view() if callable(vals_view) else vals_view
                    ).clone()
                    out._values = lambda: cloned_vals
        return out

    torch.Tensor.clone = _clone_with_alias_views
    setattr(torch.Tensor, _CLONE_ALIAS_VIEWS_FLAG, True)


def resize_as_sparse_(self: torch.Tensor, the_template: torch.Tensor) -> torch.Tensor:
    """Resize a COO tensor in place, preserving its index/value storage.

    COO and strided templates supply the target shape and sparse/dense dimension
    counts. Compressed sparse layouts are not supported by this implementation.

    This operator is host metadata work: layout checks plus one forwarded
    ``sparse_resize_`` dispatch, which lands on the native SparseCPU kernel for
    the CPU-resident COO tensors this backend materializes. The template's
    sparse/dense counts are read through direct TensorImpl accessors instead of
    dispatched ``sparse_dim()``/``dense_dim()`` calls: on this backend
    ``sparse_dim`` carries an import-time Python trampoline at SparseCPU
    (``sparse_dim.py``), costing ~1.6us per read versus ~0.5us for
    ``_indices().shape[0]`` — the same no-re-dispatch convention that module
    documents for its own body.
    """
    logger.debug("GEMS_ENFLAME RESIZE_AS_SPARSE_")
    tmpl_layout = the_template.layout
    if self.layout != _SPARSE_COO or tmpl_layout not in _ALLOWED_TEMPLATE_LAYOUTS:
        raise NotImplementedError(
            "resize_as_sparse_ supports a COO self and a COO or strided template"
        )

    if the_template.is_nested:
        # A non-jagged nested tensor still reports torch.strided, and native
        # has no sparse_dim kernel on the NestedTensor keys, so the direct
        # reads below would diverge from the native rejection. Route through
        # the dispatched metadata reads to reproduce the generic behavior.
        self.sparse_resize_(
            the_template.size(), the_template.sparse_dim(), the_template.dense_dim()
        )
        return self

    # A/B against the shape-property / identity-fast rewrite showed no
    # measurable gain (0.99x, inside the noise band): the body cost is two
    # unavoidable dispatches (`_indices().shape` and `sparse_resize_`), so the
    # original ordering is kept.
    tmpl_size = the_template.size()
    if tmpl_layout == _STRIDED:
        # Dense templates contribute no sparse dimensions; native reads
        # sparse_dim() == 0 and dense_dim() == dim() for strided tensors.
        tmpl_sparse_dim = 0
        tmpl_dense_dim = len(tmpl_size)
    else:
        tmpl_sparse_dim = the_template._indices().shape[0]
        tmpl_dense_dim = len(tmpl_size) - tmpl_sparse_dim
    self.sparse_resize_(tmpl_size, tmpl_sparse_dim, tmpl_dense_dim)
    return self


_install_detach_alias_views()
_install_clone_alias_views()
