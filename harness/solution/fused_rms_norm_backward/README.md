# fused_rms_norm_backward 修复方案

## 复现与基线
- 命令：`cd tests && pytest --timeout=900 -m fused_rms_norm_backward --ref cpu`
- 基线：24 failed / 12 passed（36 用例），失败均为编译期 MLIR 错误。
- 首个错误：generic `src/flag_gems/ops/_fused_rms_norm.py:268` `tl.sum(d_weight, axis=0)` → `axis must not be 0 for 2D+ shapes`；后续 dx kernel 又有 `triton_xpu.reduce` 类型推断错误（128 vs 256）。

## 根因
`aten::_fused_rms_norm_backward` 走 generic `flag_gems.ops._fused_rms_norm._fused_rms_norm_backward`（backend `_kunlunxin/ops/_fused_rms_norm.py` 只 override 了 forward，未 override backward）。generic 的 `_fused_rms_norm_grad_dw_kernel` 用 `ROW_BLOCK_SIZE=16` 的 2D tile 做 `tl.sum(axis=0)`，该归约在当前 XPU triton 上无法 legalize；dx kernel 的 1D `tl.sum` 也在 core-tiling 阶段类型推断失败。即 generic `_fused_rms_norm` 全套 kernel 与当前 triton 不兼容。

## 改动
在 backend `_kunlunxin/ops/_fused_rms_norm.py` 新增 `_fused_rms_norm_backward` override，复用 backend 已工作的 `.rms_norm.rms_norm_backward`（其 dw kernel 用 `ROW_BLOCK_SIZE=1`）。no-weight 情形 dx 公式 == weight=ones 结果，故传 unit weight 并丢弃 dw。在 `_kunlunxin/ops/__init__.py` 补 import（SpecOpRegistrar 按函数名替换 generic）。

关键代码：
```python
def _fused_rms_norm_backward(grad_out, input, normalized_shape, rstd, weight=None, output_mask=(True, True)):
    if len(output_mask) != 2:
        raise ValueError("output_mask must contain two booleans")
    if output_mask[1] and weight is None:
        raise RuntimeError("weight gradient requested without a weight tensor")
    if weight is None:
        if not output_mask[0]:
            return None, None
        n = math.prod(normalized_shape)
        unit_weight = torch.ones(n, dtype=input.dtype, device=input.device)
        dx, _ = rms_norm_backward(grad_out, input, rstd, normalized_shape, unit_weight)
        return dx, None
    dx, dw = rms_norm_backward(grad_out, input, rstd, normalized_shape, weight)
    return (dx if output_mask[0] else None), (dw if output_mask[1] else None)
```

## 验证
- `pytest -m fused_rms_norm_backward --ref cpu`：**36 passed**（原 24 failed / 12 passed）。
- dispatch 校验：`flag_gems._fused_rms_norm_backward.__module__` == `_kunlunxin.ops._fused_rms_norm`。
- `py_compile` 通过。

## 遗留
- 复用 `rms_norm_backward` 内 `torch.sum(partial_buffer, dim=0)` 做跨行最终归约——与 backend 既有 `rms_norm_backward` 完全一致，属已接受模式（重活由 XPU kernel 完成，torch.sum 只做 partial buffer 的最终小归约）。
- 尚未做 3× fresh-cache / benchmark / 收尾 AB（性能口径见 TODO）。
