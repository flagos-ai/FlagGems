# fractional_max_pool3d_backward 修复方案（完成）

## 复现与基线
- 命令：`cd tests && pytest --timeout=900 -m fractional_max_pool3d_backward --ref cpu`
- 基线：18 failed（全部用例），编译期 MLIR 错误。
- 首个错误：`ConvertTritonXPUToLLVM` 失败，`tl.atomic_add`（fp32 累加也一样）带 `atomicSim=true` 无法 lower。XPU 上原子 scatter 本身不可用。

## 根因
1. `native_atomic` 判定错误：`grad_output.is_cuda` 在 XPU 上为 True（torch_xmlir 伪装 cuda）、`get_device_capability()` 返回伪装 sm_86，fp16 会误走 native fp16 `atomic_add`。已加 `flag_gems.vendor_name == "nvidia"` 门禁（必要且正确，保留）。
2. 更深层：即使切到 fp32 累加，`tl.atomic_add` 的 `atomicSim` 在 `ConvertTritonXPUToLLVM` 仍无法 lower——XPU 上原子 scatter 本身不可用（backend `fractional_max_pool2d` 注释已证实，其用非原子 gather/scatter）。

## 改动（`src/flag_gems/ops/fractional_max_pool3d_backward.py`，generic 文件）
1. 新增 `_fractional_max_pool3d_backward_gather_kernel`（非原子 gather，仿 backend `fractional_max_pool2d.py` 的 `_fractional_max_pool2d_backward_gather_kernel`，把 2D 推广到 3D）：
   - **grid/索引**：一 lane 一输入位置（flat 覆盖 `N*C*D*H*W`），`offsets -> nc = offsets//in_dhw, d/h/w = rem 分解`，本 lane 的 plane-local flat 索引 `my_flat = d*in_hw + h*in_w + w`。
   - **窗口遍历**：对每个输入坐标反推可能映射到它的输出窗口，`od/oh/ow ∈ [max(0, trunc((x-k)*inv_alpha)-2), min(out-1, trunc((x+1)*inv_alpha)+2)]`，三重 `range(0, MAX_*)` 滚动循环；每个候选输出 `o_off` 读 `indices[o_off]` 与 `grad_output[o_off]`，`idx == my_flat` 时把 grad 累加进 fp32 寄存器；循环结束单次 masked store。
   - `inv_alpha = 1/alpha`（`out==1` 或 `in==k` 时为 0）；`MAX_* = min(out, ceil((k+1)/alpha)+6)`，覆盖所有候选并吸收 fp32 舍入；候选被存储 indices 精确验证，等价于全量扫描、确定性、无竞争。
2. host 函数：`native_atomic`（真实 NVIDIA）时保留原 autotuned 原子 scatter 快路径；否则（XPU 等）走 gather kernel（fp32 累加，最后 `.to(grad_output.dtype)` 回原 dtype）。空张量提前返回 `zeros_like`。
3. `import math` 用于 host 端 `MAX_*` 计算；`import flag_gems` 用于 vendor 门禁（此前已有）。

## 测试侧改动（`tests/test_fractional_max_pool3d_backward.py`）
- 生成 `res_indices` 的 forward 由 `torch.nn.functional.fractional_max_pool3d` 改为 `flag_gems.fractional_max_pool3d`。原因：torch native XPU forward 对 bf16（input 与 random_samples 均 bf16）输出**错误的值与索引**（valdiff≈2.98、idxdiff 数百），而 `flag_gems.fractional_max_pool3d` 与 torch CPU fp64 参考在所有 dtype/shape 下**完全一致**（worst idxdiff=0/valdiff=0，6 shape × 3 dtype × 5 seed 验证）。backward 收到的应是 flag_gems 实际 forward 产生的 indices；注释里的 "forward is a separate PR" 已过时（forward 已实现并修好，见 harness/solution/fractional_max_pool3d/）。

## 验证
- 功能：`pytest --timeout=900 -m fractional_max_pool3d_backward --ref cpu -q` = **18 passed**。
- 3× 独立进程 + 独立 `FLAGGEMS_CACHE_DIR`（/tmp/pool3db_run{1,2,3}）均为 **18 passed**。
- benchmark：`cd benchmark && pytest -m fractional_max_pool3d_backward -q` = **1 passed**（无编译/运行回退）。
- `python -m py_compile` 两个改动文件 OK。
- fallback 静态检查：无 `[FAIL]`（仅 `torch.empty_like`/`torch.zeros_like`/`torch.zeros` 内存分配与 `.to()` 数据搬运，不承担被测数值计算）。

## 遗留
- 属 generic 文件改动（跨后端语义等价；NVIDIA 路径保留原 atomic scatter，gather 仅用于非 native-atomic 后端）。
- XPU 上 gather 是通用正确路径，但重叠窗口下的三重 `MAX_D×MAX_H×MAX_W` 滚动循环比 scatter 慢；测试矩阵（kernel=2、alpha=2、非重叠）下纯拷贝，无性能回归风险。后续如需性能优化可在 alpha>=k 时加 disjoint scatter 快路径（对齐 pool2d 的 scatter 快路径）。
- 测试运行时的 `XDNN_UNIMPLEMENTED ... ceil.cpp` 报错为环境既有噪声（forward 测试同样出现且 18 passed），非本改动引入。
