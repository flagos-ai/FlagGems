# dirichlet_grad 修复方案

## 基线
- `pytest -m dirichlet_grad --ref cpu`：48 failed / 48 passed（仅 float64 用例失败）。

## 根因
XPU 无 fp64（`flag_gems.runtime.device.support_fp64` 为 False，torch 静默把 float64 张量降级 fp32），`_dirichlet_grad` 的 float64 用例在 fp32 下计算、再与 fp64 CPU reference 对拍必然失败。这是**设备不支持**（SKIP），非算子 bug——backend `_dirichlet_grad.py` 已明确 fp32 路径正确。

## 改动（测试侧，属设备限制门控）
`tests/test_dirichlet_grad.py`：
```python
DIRICHLET_GRAD_DTYPES = (
    [torch.float32, torch.float64] if utils.fp64_is_supported else [torch.float32]
)
```

## 验证
- `pytest -m dirichlet_grad --ref cpu`：**48 passed**（fp32 全过，fp64 不再收集）。

## 说明
按 AGENTS.md §5 归类为 SKIP（设备不支持 fp64），不加入 PASS 回归清单的 fp64 部分。
