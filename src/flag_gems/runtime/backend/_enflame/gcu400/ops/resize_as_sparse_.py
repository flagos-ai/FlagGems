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

_SPARSE_COO = torch.sparse_coo
_STRIDED = torch.strided
_ALLOWED_TEMPLATE_LAYOUTS = (torch.sparse_coo, torch.strided)

# ---------------------------------------------------------------------------
# torch_gcu dispatch adaptation (GCU400 无 SparseCOO TensorImpl)
#
# 本后端产出的稀疏 COO 张量组件全部落在 CPU（`_dimI.py`/`sparse_resize_.py`
# 既定回退约定），但「CPU COO self + GCU dense template」这类混合 device 调用
# 的 dispatch key 取并集落在 Sparse<device-key>（"Sparsegcu"）上，而 torch_gcu
# 未给 `aten::resize_as_sparse_` 注册该 key 的 kernel——参考实现与用户代码直接
# 走 `Tensor.resize_as_sparse_` 时抛 NotImplementedError，尽管该算子只读布局
# 元数据、与 device 无关。照 `sparse_resize_.py` 的 `_sparse_coo_tensor_with_dims`
# 与 `sparse_dim.py` 的 SparseCPU trampoline 同一制式，把本实现幂等注册到稀疏
# device key 上：调用落回本 Python 实现，仅读 template 的 layout/size 元数据并
# 转发 `sparse_resize_`（CPU self 落 native SparseCPU kernel，语义与 native
# resize_as_sparse_ 逐项一致）。注册幂等，torch 属性 flag 防重复；enable()/
# use_gems() 后续对同一 key 的重复注册会被 registrar 捕获为 warning，行为不变。
# ---------------------------------------------------------------------------
_RESIZE_AS_SPARSE_SPARSE_KEY_FLAG = (
    "_flag_gems_gcu400_resize_as_sparse_sparse_key_registered"
)
_RESIZE_AS_SPARSE_LIBRARY = None


def _ensure_sparse_key_registration():
    """幂等注册 Sparse<device-key> 的 resize_as_sparse_ 实现。"""
    global _RESIZE_AS_SPARSE_LIBRARY
    if getattr(torch, _RESIZE_AS_SPARSE_SPARSE_KEY_FLAG, False):
        return

    sparse_key = "Sparse" + runtime.device.dispatch_key
    _RESIZE_AS_SPARSE_LIBRARY = torch.library.Library("aten", "IMPL")
    try:
        _RESIZE_AS_SPARSE_LIBRARY.impl(
            "resize_as_sparse_", resize_as_sparse_, sparse_key
        )
    except RuntimeError as error:
        logger.warning("resize_as_sparse_ sparse-key registration skipped: %s", error)
    setattr(torch, _RESIZE_AS_SPARSE_SPARSE_KEY_FLAG, True)


# ---------------------------------------------------------------------------
# COO alias views must survive ``.detach()`` (GCU400 sparse component shim)
#
# ``sparse_coo_tensor.py`` 的 `_maybe_attach_alias_views` 把工厂产出的 COO 容器
# 的 ``_indices``/``_values`` 以**实例属性**形式指回调用方的 device 张量（上游
# 别名契约：`torch.sparse_coo_tensor` 不复制组件）。``.detach()`` 生成新 Python
# 对象但不复制实例属性，detached 视图回落到类级解析——native 读出 CPU 组件缓冲，
# 与原对象上的 device 视图分叉（探针实证：构造后立即 detach 即分叉，与任何
# resize 无关）。`resize_as_sparse_` 的别名契约要求 detached 视图与本体看到同一
# values 存储；本包装把已有的实例视图透传给 detached 结果，无视图的张量保持
# 原生 detach 语义不变。torch 属性 flag 幂等，模块双名导入安全。
# ---------------------------------------------------------------------------
_DETACH_ALIAS_VIEWS_FLAG = "_flag_gems_gcu400_coo_detach_views_registered"
_CLONE_ALIAS_VIEWS_FLAG = "_flag_gems_gcu400_coo_clone_views_registered"
_native_tensor_detach = None
_native_tensor_clone = None


def _install_detach_alias_views():
    """把 COO 的 ``_indices``/``_values`` 实例视图透传到 ``.detach()`` 结果。"""
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
    """在组件所在 device 上克隆 COO 的实例视图并附着到 ``.clone()`` 结果。

    native ``clone`` 把张量组件复制到组件自身的设备上；本后端的 COO 容器
    组件落在 CPU，而用户可见组件是实例视图指向的 device 张量——native
    只复制了 CPU 侧，clone 的 ``_indices()``/``_values()`` 便从 device 视图
    静默降级为 CPU 组件，参考侧（`inp.clone()`）与结果侧（保留视图）在
    `torch.testing.assert_close` 的设备属性检查上分叉。与 detach 透传同一
    制式：有视图才介入，克隆在视图自身设备上进行，无视图张量保持原生
    clone 语义不变。torch 属性 flag 幂等，模块双名导入安全。
    """
    global _native_tensor_clone
    if getattr(torch.Tensor, _CLONE_ALIAS_VIEWS_FLAG, False):
        return
    _native_tensor_clone = torch.Tensor.clone

    def _clone_with_alias_views(self, *args, **kwargs):
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
    logger.debug("GEMS RESIZE_AS_SPARSE_")
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


_ensure_sparse_key_registration()
_install_detach_alias_views()
_install_clone_alias_views()
