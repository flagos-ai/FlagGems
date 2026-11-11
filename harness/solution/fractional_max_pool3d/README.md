# fractional_max_pool3d 修复方案（功能 + 性能）

## 复现与基线
- 功能：`cd tests && pytest --timeout=900 -m fractional_max_pool3d --ref cpu`，基线 18 failed（编译期 MLIR 错误，见下）。
- 性能：`cd benchmark && pytest -m fractional_max_pool3d -s`，基线 dtype 等权 Gems Speedup ≈ **0.714**（fp16 0.740 / fp32 0.743 / bf16 0.659）。

## 根因

### 功能（此前已修）
generic kernel 累加器 `tl.full(..., dtype=<tl.dtype对象>)` 在 XPU 产出 f32 而 load 得 bf16，`scf.for` init/iter 类型不一致导致编译失败。已改为 f32 累加 + store cast（详见此前记录）。

### 性能（本轮）
pre-profiling（`harness/perf_ir/prof_pool3d_pre.log`）显示 forward kernel 占 ~99% 时间：大 output（1,64,8,56,56→4,28,28）单次 ~38ms（do_bench 实测 11.1ms），小 output（8,16,4,8,8→2,4,4）~23.6ms。瓶颈不在累加器 dtype，而在 kernel 结构：

1. `@triton.autotune` 6 个 config，逐 shape 重编译（IR 里 6 份 kernel 变体）。
2. 2D-tile 内三重 `for kd/kh/kw` 展开的 **masked load**（`other=运行时 min_val`）+ `tl.where` 逐元素比较，走上慢 masked-memory 路径（知识库 §3.4）。
3. **int64 索引累加器**：循环内 i32→i64 2 倍宽 cast + 每迭代 int64 select（知识库 item27/43 反模式）。
4. 基准/测试全部是 `output=half spatial, kernel=2`，`alpha=(in-k)/(out-1)==2==kernel` 恰为整数 → `start(i)=2i`，fractional 退化为**规则 stride-2 max pool**；此时逐 lane stride-2 gather 是可向量化访问，但默认 codegen 没开 vectorization，每元素 ~5ns 的 gather 是硬墙。

## 改动（`src/flag_gems/ops/fractional_max_pool3d.py`，generic 文件）

1. **删 `@triton.autotune` + 6 config**，改为**单 config 的 1D-flat kernel**（对齐 backend 已工作的 `fractional_max_pool2d` 模式）：每个 lane 对应一个输出元素，grid = cdiv(numel, BLOCK_SIZE)。
2. **去 masked load**：host assert `out+k-1<=in` 保证每个合法输出的池化窗都在界内（alpha>=1），因此 load 无需 mask；tail lane 用 `safe_offsets = tl.where(store_mask, offsets, 0)` 折叠到合法位置，只在 store 上带 mask。
3. **int32 索引累加器**（host 传 `USE_INT32_IDX = (in_d*in_h*in_w) < 2**31`），store 时 `.to(tl.int64)`；巨大平面回退 int64。
4. 累加器保持**输入 dtype**（fp16/bf16/fp32 不升 f32）。
5. **XPU codegen knobs**（对其他后端被 `parse_options` 静默忽略）：`isCloseVectorization=True, buffer_size_limit=2048, num_warps=1` —— 这正是让 backend 向量化 stride-2 gather 的关键（`fractional_max_pool2d` 同款）。

尝试过但否决：regular-stride 专用「连续 load + `tl.reshape`/`tl.split` 寄存器成对归约」fast path —— 在 XPU `ConvertTritonToTritonXPU`/`uni_sram` 无法 lower（OOM），已放弃。

## 验证
- 功能：`pytest --timeout=900 -m fractional_max_pool3d --ref cpu -q` = **18 passed**；3× 独立进程 + 独立 `FLAGGEMS_CACHE_DIR` 均为 18 passed。
- `python -m py_compile` OK；fallback 静态检查无 `[FAIL]`（仅 `torch.empty`/`torch.rand` 内存分配/随机采样，非数值计算）。
- benchmark 全矩阵 SUCCESS，dtype 等权 Gems Speedup：

| dtype | 前 (mean) | 后 (mean) |
|---|---|---|
| float16 | 0.740 | 2.998 |
| float32 | 0.743 | 3.282 |
| bfloat16 | 0.659 | 2.719 |
| **dtype 等权** | **0.714** | **3.000** |

- 最差 case（1,64,8,56,56→4,28,28）：11.11ms → 2.83ms（speedup 0.13 → 0.54~0.61），仍低于 1.0 但不再是灾难。
- profiling 前后：forward kernel 大 output 38.3ms → 2.88ms（×13.3），小 output 23.6ms → 0.07ms；kernel 调用次数不变（30），无拆核伪优化。

## 遗留
- 属 generic 文件改动（影响所有后端；1D-flat 单 config + 去 mask + int32 索引在所有后端语义等价，XPU codegen knobs 在 NVIDIA 上被忽略）。
- 最差 case 仍是 stride-2 gather 墙（torch native ~1.5ms vs gems ~2.8ms）；进一步需连续块 DMA 方案，但 reshape/split 在 XPU 无法 lower，暂未突破。
- 尚未跑收尾 A/B（git stash 对比）；改动后工作区即 B 态。
