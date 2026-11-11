# fft_rfft 功能修复方案（Kunlunxin / XPU）

## 1. 复现与基线

- 入口：`cd tests && pytest --timeout=900 -m fft_rfft --ref cpu -q`
- 基线（修复前）：`29 failed / 14 passed / 3 skipped`
- 首个真实错误（全部 29 个失败同源）：
  ```
  triton.runtime.errors.OutOfResources: out of resource: uni_sram
  PassManager::run failed ... Pipeline failed while executing [`TritonXPULegalize`]
  UNREACHABLE executed at .../TritonXPU/Transforms/Legalize.cpp:71
  ```
- 失败触发点：`src/flag_gems/ops/fft_rfft.py` 中两个 kernel 的循环内
  `acc_re += tl.sum(xm * tl.cos(ang), axis=1)`（旧 83/216 行附近）。

## 2. 根因

- 旧实现是「每个 program 负责 BLOCK_K 个输出频点、循环内对 `[BLOCK_K, BLOCK_J]`
  相位矩阵做 2D 轴-1 归约，再累加到 1D 累加器」：
  ```python
  m   = (j[None, :] * keff[:, None]) % n      # [BLOCK_K, BLOCK_J] i32
  ang = m.to(tl.float32) * step              # [BLOCK_K, BLOCK_J] f32
  acc_re += tl.sum(xm * tl.cos(ang), axis=1) # 2D 归约 -> [BLOCK_K]
  acc_im -= tl.sum(xm * tl.sin(ang), axis=1)
  ```
- 该结构在 Kunlunxin XPU 后端 lowering 失败（`TritonXPULegalize` 中 `uni_sram`
  out-of-resource，`getSlicedType` 对 2D 张量 slice 时 tensor rank > 归约树 iterCount
  触发 `llvm_unreachable`）。我尝试过“persisted 2D 累加器 + 循环后单次
  `tl.sum(axis=1)`”（对齐 backend `mv.py` 的 proven 模式），但循环体内
  `m/ang/cos/sin` 等 2D 中间张量 + 双 2D 累加器仍超出 uni_sram，依旧 OOR。
- 依据知识库 `KERNEL_OPT_EXPERIENCE.md` item 2/4/12/37 与 `HARNESS_SUMMARY.md` §3.5/§4：
  XPU 上 2D 轴-1 归约（尤其循环内）不可靠；1D 循环携带累加器在 lane ≤64 时安全。

## 3. 修复（最小改动，纯 Triton，无 fallback）

把两个 DFT kernel 从「BLOCK_K 个频点/程序 + 2D 归约」改为「**每个 (batch row, 频点) 一个
program + 1D 归约**」：

- 去掉 `BLOCK_K` 维度：`r = pid // K; k = pid % K`（`k` 为标量频点索引）。
- 循环内相位改为 1D：`m = (j * k) % n`，`acc_re += xj * tl.cos(ang)`，
  `acc_im -= xj * tl.sin(ang)`（1D `[BLOCK_J]` 累加）。
- 循环后单次 1D→标量归约：`re = tl.sum(acc_re); im = tl.sum(acc_im)`。
- DC/Nyquist 虚部清零、norm 缩放、标量 store 均改为标量语义。
- host 侧：`blk_j = max(16, min(64, _next_pow2(nn)))`（1D tile 卡到 ≤64 lane，
  知识库安全点）；grid 从 `R * nk` 改为 `R * K`。

修改文件：`src/flag_gems/ops/fft_rfft.py`（**generic**；用户明确指示「优先改 generic
使其 XPU 兼容」，与 harness 默认「只改 `_kunlunxin/`」存在偏差，已按用户指示执行并在此记录）。
无任何 torch FFT / torch 计算算子 fallback，数值计算全部在 Triton kernel 内完成。

## 4. 正确性验证

- 最小失败节点：`test_fft_rfft_transform_length[16]` → `1 passed`。
- 完整 marker（3 次独立进程 + 3 个独立 `FLAGGEMS_CACHE_DIR` fresh-cache）：
  ```
  RUN1: 42 passed, 3 skipped, 1 failed
  RUN2: 42 passed, 3 skipped, 1 failed
  RUN3: 42 passed, 3 skipped, 1 failed
  ```
  唯一失败 `test_fft_rfft_defaults_bind` 为 **NON_BUG**（见下）。
- 独立探针验证 `defaults_bind` 语义正确（`fft_rfft(inp)` 与
  `fft_rfft(inp, n=None, dim=-1, norm=None)` 结果逐位相等）：
  ```
  equal_cpu True, max_abs_diff 0.0
  ```
- benchmark：`cd benchmark && pytest -m fft_rfft -q` → `1 passed`（无编译/运行回退）。

## 5. 遗留问题（NON_BUG）

`test_fft_rfft_defaults_bind` 失败原因与算子无关：该用例断言
`torch.equal(flag_gems.fft_rfft(inp), flag_gems.fft_rfft(inp, n=None, dim=-1, norm=None))`，
而 `torch.equal` 对 complex64 张量在 XPU 上未实现：

```
[XDNN_UNIMPLEMENTED] .../equal.cpp:66 scalar type of self : kcomplex64 is unsupported in xdnn_pytorch_wrapper !
RuntimeError: [NOT IMPLEMENTED]: error code= 4 (.../equal.cpp:44)
```

- 分类：**参考侧/测试环境问题（NON_BUG）**，非 flag_gems kernel bug。
- 证据：`torch.equal(a.cpu(), b.cpu()) == True`，`max_abs_diff == 0.0`；device 端
  `torch.equal` 直接抛 NOT IMPLEMENTED。
- 处置建议（需人类决策，不擅自改测试）：可将该用例比较方式改为
  `torch.allclose(...)` 或分别比较 `.real/.imag`，或加 `skipif(vendor_name=="kunlunxin")`；
  本方案未修改 `tests/`。

## 6. fallback / 编译门禁

- fallback 静态检查：本次改动不在 `src/flag_gems/runtime/backend/_kunlunxin/`，脚本扫描
  范围为 backend 文件，无命中；人工核查 kernel 全部为 `tl.*`，host 仅 `torch.empty` /
  `torch.as_tensor` / `torch.view_as_complex`（内存分配/视图，非计算 fallback）。
- `python -m py_compile src/flag_gems/ops/fft_rfft.py` → OK。
- 未 commit / 未 push。
