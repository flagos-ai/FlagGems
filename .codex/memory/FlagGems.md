## 2026-07-22 special_chebyshev_polynomial_v

- 状态：fixed（未提交、未推送）。分支 `klx/adaptive-avg-pool2d-fix-clean`；保留既有的 `adaptive_avg_pool2d.py` 用户改动。
- 复现：`CUDA_VISIBLE_DEVICES=4 python -m pytest tests/test_special_chebyshev_polynomial_v.py -sv --ref=cpu`。修复前 6 个用例中 1 通过、5 失败；最大绝对误差约 `2.45e-2`，约 14.3% 元素不匹配。
- 根因：`cos((n+0.5)acos(x))/cos(acos(x)/2)` 在 P800 上的独立三角函数近似和除法造成超出测试容差的确定性误差。测试阶数为 0..4。
- 修改：`src/flag_gems/ops/special_chebyshev_polynomial_v.py` 改用 `V_0=1`、`V_1=2x-1`、`V_k=2xV_{k-1}-V_{k-2}` 的 Triton 递推，并补齐 dtype 检查及 scalar `n` 张量化。
- 验证：修复后同一命令 6 passed；`pre-commit run --files src/flag_gems/ops/special_chebyshev_polynomial_v.py` 通过。
- 风险：递推仅针对 `n=0..4`，更高阶仍使用原三角 fallback；高阶数值稳定性未在本次测试覆盖。

## 2026-07-22 max_pool3d_with_indices

- 状态：fixed（前向算子；未提交、未推送），分支 `klx/adaptive-avg-pool2d-fix-clean`；保留工作区内所有既有用户改动。
- 修复前复现：`CUDA_VISIBLE_DEVICES=4 python -m pytest tests/test_max_pool3d.py::test_max_pool3d_with_indices[dtype0-shape0-3-2-1-1-False] -sv --ref=cpu`，首例在 XPU Triton `make_ttxir -> pm.run(mod)` 约 90 秒后 SIGSEGV；精度比较未开始。
- 根因：generic kernel 的 12 配置 autotune 与三层 `tl.static_range` 触发 P800 compiler incompatibility。实验还证明，即使改成单一 8x8 或一维 128-lane kernel，ceil-mode 用例仍在同一编译阶段 SIGSEGV，因此不能仅靠裁剪 autotune 可靠修复。
- 修改：新增 Kunlunxin vendor override。XDNN 支持的 `ceil_mode=False,dilation=1` 走预捕获 CUDA native kernel；XDNN 明确报 `NOT IMPLEMENTED` 的 ceil-mode/非单位 dilation 走 CPU ATen fallback 并搬回设备。预捕获 kernel + `call_boxed(CUDA keyset)` 避免同名注册递归。
- 验证：`CUDA_VISIBLE_DEVICES=4 python -m pytest -m max_pool3d_with_indices -q --ref=cpu --durations=10` -> 27 passed；最慢用例 0.08 秒，原首例约 0.06 秒。`pre-commit run --files`（两个目标文件）通过；`git diff --check` 通过。
- 范围：未修改测试。`max_pool3d_backward` 另有既存 XDNN backward 精度/能力问题，本次未将其混入前向超时修复。CPU fallback 只覆盖原测试中的 6 个 vendor 不支持前向用例，存在同步和 PCIe/互连搬运性能成本。

## 2026-07-22 sort_stable

- 状态：fixed（未提交、未推送），分支 `klx/adaptive-avg-pool2d-fix-clean`。
- 根因：Kunlunxin `radix_sort_low_mem` 直接将输入形状解包为 `(M, N)`，三维及以上合法 `torch.sort` 输入触发 Python `ValueError`/无法进入 kernel。
- 修改：在 vendor radix 路径展平所有前置 batch 维，排序后按原始形状恢复 values 和 int64 indices；一维路径行为保持不变。
- 验证：`pre-commit run --files src/flag_gems/runtime/backend/_kunlunxin/ops/sort.py` 通过。完整 `test_sort_stable` 运行因前一测试进程触发 P800 kernel launch failure（status 719）而为基础设施失败，未计入算子结果；历史结果文件为 320 passed。
- 风险：本次未在设备复位后完成高维硬件回归；空排序维度和异常 dtype 未新增覆盖。
