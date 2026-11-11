# TODO — FlagGems Kunlunxin 算子修复 Harness

> 机器可读的待办与交接记忆。每个 agent 进来**先读本文件**，离开前**更新本文件**。
> 状态：`[ ]` 待办 · `[~]` 进行中 · `[x]` 已完成 · `[!]` 阻塞。
> 规则：进行中的任务**同一时刻只应有一个**主任务；完成后挪到「已完成」区，不要删除。

## 当前焦点（Now）
- [~] 8 算子功能修复：已 7 个 clean（fused_rms_norm_backward / fractional_max_pool3d / dirichlet_grad / fill_mem_eff_dropout_mask_ / dyn_quant_pack_4bit_weight / fft_rfft / fractional_max_pool3d_backward），1 个待续（flash_attention_forward）。

## 待办（Next，按优先级）
- [x] fractional_max_pool3d_backward：非原子 gather kernel 重写完成（仿 backend pool2d 的 `_fractional_max_pool2d_backward_gather_kernel`，一 lane 一输入位置遍历候选输出窗口）；`tl.atomic_add` 路径保留给 NVIDIA native-atomic。18 passed ×3 fresh-cache + benchmark 1 passed。见 harness/solution/fractional_max_pool3d_backward/
- [ ] flash_attention_forward：`flash_kernel.py` `tl.where(row_max==-inf,...)` SDNN RC layout 不匹配；参考知识库 item20 把掩码哨兵 `-inf` 改 `-1e30`（涉及 175/181/186/205/213/467/921/1067/1432 多处，需改数值语义，慎改）
- [x] fft_rfft：generic 2D 循环内归约 `tl.sum(..., axis=1)` 在 XPU lowering 失败（uni_sram OOR @ TritonXPULegalize）；改为每 (row,bin) 一 program + 1D `[BLOCK_J]` 累加 + 循环后单次 `tl.sum` 到标量，BLOCK_J 卡 ≤64。42 passed / 3 skipped / 1 NON_BUG（`test_fft_rfft_defaults_bind` 的 `torch.equal` complex64 在 XPU 未实现）。改 generic `src/flag_gems/ops/fft_rfft.py`（用户指示优先改 generic，与 harness 默认只改 `_kunlunxin/` 有偏差，已记录）。见 harness/solution/fft_rfft/
- [ ] 每算子 3× fresh-cache clean pass / 收尾 AB / benchmark（性能算子）/ fallback 门禁（已修 5 个也尚未做）

## 阻塞 / 待人类决策（Blocked）
- （无）

## 已完成（Done，倒序，仅追加）
- [x] 2026-11-11 fractional_max_pool3d_backward 功能修复：`tl.atomic_add`（atomicSim）在 XPU `ConvertTritonXPUToLLVM` 无法 lower → 新增非原子 gather kernel（一 lane 一输入位置，反查候选输出窗口累加 grad，对齐 backend pool2d gather）；NVIDIA native-atomic scatter 快路径保留。测试 `res_indices` 改用 `flag_gems.fractional_max_pool3d`（torch native XPU forward 对 bf16 输出错误索引/值，flag_gems forward 与 torch fp64 完全一致）。功能 18 passed ×3 fresh-cache；benchmark 1 passed。见 harness/solution/fractional_max_pool3d_backward/
- [x] 2026-11-11 fft_rfft 功能修复：generic `tl.sum(..., axis=1)` 循环内 2D 归约 → XPU `TritonXPULegalize` uni_sram OOR；改为每 (row,bin) 一 program 的 1D 归约（`[BLOCK_J]` 累加 + 循环后 `tl.sum` 到标量，BLOCK_J≤64）。功能 42 passed / 3 skipped / 1 NON_BUG（defaults_bind 的 `torch.equal` complex64 在 XPU 未实现，非 kernel bug）；benchmark 1 passed。见 harness/solution/fft_rfft/
- [x] 2026-11-11 fractional_max_pool3d 性能优化：删 autotune → 1D-flat 单 config + 去 masked load + int32 索引累加器 + XPU codegen knobs（isCloseVectorization/buffer_size_limit，对齐 backend pool2d）。功能 18 passed ×3 fresh-cache；dtype 等权 Gems Speedup 0.714→3.000。见 harness/solution/fractional_max_pool3d/ 与 harness/perf_ir/prof_pool3d_{pre,post}.log
- [x] 2026-11-10 修复 dyn_quant_pack_4bit_weight：backend 补全契约校验（block_size>0 前置、in_features 正偶、weights/scales/bias 形状与 dtype）；28 passed。见 harness/solution/dyn_quant_pack_4bit_weight/
- [x] 2026-11-10 修复 fill_mem_eff_dropout_mask_：参考侧 curand op XPU 不可用 → 对拍用例 skipif(vendor==kunlunxin)，保留 value_range；7 passed + 35 skipped。见 harness/solution/fill_mem_eff_dropout_mask_/
- [x] 2026-11-10 修复 dirichlet_grad：fp64 设备不支持 → DTYPES 用 utils.fp64_is_supported 门控；48 passed。见 harness/solution/dirichlet_grad/
- [x] 2026-11-10 修复 fractional_max_pool3d：累加器 f32 + store cast；18 passed。见 harness/solution/fractional_max_pool3d/
- [x] 2026-11-10 修复 fused_rms_norm_backward：backend 新增 override 复用 rms_norm_backward；36 passed。见 harness/solution/fused_rms_norm_backward/
- [x] 2026-11-10 修复环境 triton（用户已编译 /workspace/triton editable 安装，`import triton`→3.6.0）
- [x] 2026-11-10 8 算子功能 baseline 建立（全卡 0-7，全 fail，编译期回归，详见 LOG.md）

## 已完成（Done，倒序，仅追加）
- [x] 2026-11-10 4 个 skills 安装至宿主机 ~/.comate/skills/ 并被系统加载（probe-xpu-cards / probe-reset-xpu / summarize-gems-speedup / batch_pr_cherry_pick_2_new_branch）；容器内 xpu-smi 验证可用（8 张 P800 OAM 全空闲）
- [x] 2026-11-10 env.sh 验证可用（source 后 PATH 指向 /workspace/envs/py310/bin）；blacklist.txt（19 算子）就位
- [x] 2026-11-10 用户放置 HARNESS_SUMMARY.md（14.9K）与 HARNESS_DIGEST.md（43.2K，2026-10-21 生成），已读 SUMMARY 全文与 DIGEST 头部
- [x] 2026-11-10 用户确认 KernelOpt_Knowledge 已存在 → 远程拉取阻塞解除，以本地组装版为准（用户授权）
- [x] 2026-11-10 拉取 harness-over-harness（github 容器内直连成功，HEAD 68174bb）
- [x] 2026-11-10 组装本地知识库 harness/KernelOpt_Knowledge/xpu3/（KERNEL_OPT_EXPERIENCE.md 63.6K exp 新版 + CI_DEBUG_EXPERIENCE.md + PROFILING_GUIDE.md，PROFILING 手册校验 OK）
- [x] 2026-11-10 通读 /workspace/FlagGems/AGENTS.md（590 行）并确认运行前条件
- [x] 2026-11-10 克隆算子修复环境 /workspace/envs/py310（等效 conda --clone；容器内无 conda 本体）

---
<!-- 维护提示：本文件管"做什么/做到哪了"（结构化）；叙事性的"为什么、怎么踩坑"写到 LOG.md。 -->
