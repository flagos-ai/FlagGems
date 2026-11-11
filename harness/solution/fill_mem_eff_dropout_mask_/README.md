# fill_mem_eff_dropout_mask_ 修复方案

## 基线
- `pytest -m fill_mem_eff_dropout_mask_`：35 failed / 7 passed。

## 根因
flag_gems 的 `_fill_mem_eff_dropout_mask_` kernel 本身工作正常（`test_..._value_range` 7 例全过，输出在 [0,1)）。失败的两类用例都做**与 ATen reference 的逐位对拍**：`torch.ops.aten._fill_mem_eff_dropout_mask_`（curand 实现）在 XPU 上无法 eager dispatch（torch_xmlir 只注册为 lazy/backend fallback，`NotImplementedError: ... CPU backend`）。这是**参考侧不可用**（NON_BUG），非算子 bug。

## 改动（测试侧，参考不可用门控）
`tests/test_fill_mem_eff_dropout_mask_.py` 对两个 reference 对拍用例加：
```python
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="aten::_fill_mem_eff_dropout_mask_ (curand) reference unavailable on XPU",
)
```
保留 `test_..._value_range`（不需要 reference）继续跑。

## 验证
- `pytest -m fill_mem_eff_dropout_mask_`：**7 passed + 35 skipped**（原 35 failed / 7 passed）。

## 说明
归类为 NON_BUG（参考侧 XPU 不可用）。XPU 上无法验证与 curand 的逐位一致性，只能验证值域；保留 value_range 用例覆盖 kernel 基本正确性。
