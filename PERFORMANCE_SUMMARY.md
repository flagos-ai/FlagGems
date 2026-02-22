# FlagGems 操作性能基准测试结果

## 测试环境
- GPU: CUDA设备
- 测试模式: kernel mode
- 测试级别: core
- Warmup: 50 iterations
- 测试迭代: 100 iterations

## 性能结果汇总

### 1. cosh (已测试)

**分支**: `codex/cosh`

| 数据类型 | Shape | PyTorch延迟(ms) | FlagGems延迟(ms) | 加速比 |
|---------|-------|----------------|------------------|--------|
| float16 | [1073741824] | 4.679 | 4.882 | 0.958 |
| float16 | [64, 64] | 0.004 | 0.004 | 1.000 |
| float16 | [4096, 4096] | 0.079 | 0.077 | 1.027 |
| float16 | [64, 512, 512] | 0.079 | 0.078 | 1.013 |
| float16 | [1024, 1024, 1024] | 4.669 | 4.895 | 0.954 |
| float32 | [1073741824] | 9.349 | 9.629 | 0.971 |
| float32 | [64, 64] | 0.004 | 0.003 | 1.306 |
| float32 | [4096, 4096] | 0.155 | 0.152 | 1.020 |
| float32 | [64, 512, 512] | 0.154 | 0.151 | 1.020 |
| float32 | [1024, 1024, 1024] | 9.345 | 9.652 | 0.968 |

**平均加速比**: 
- float16: ~1.0x (与PyTorch相当)
- float32: ~1.0x (与PyTorch相当)

**结论**: cosh 优化后性能与 PyTorch 基线相当，在小尺寸输入上略有优势。

---

### 2. max_pool3d

**分支**: `codex/max_pool3d`

**状态**: 需要运行基准测试

**测试命令**:
```bash
pytest benchmark/test_reduction_perf.py -m max_pool3d -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float16 --dtypes float32 --warmup 50 --iter 100
```

**预期**: 需要实际测试结果

---

### 3. avg_pool3d

**分支**: `codex/avg_pool3d`

**状态**: 需要运行基准测试

**测试命令**:
```bash
pytest benchmark/test_reduction_perf.py -m avg_pool3d -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float16 --dtypes float32 --warmup 50 --iter 100
```

**预期**: 需要实际测试结果

---

### 4. grid_sample

**分支**: `codex/grid_sample`

**状态**: 需要运行基准测试

**测试命令**:
```bash
pytest benchmark/test_special_perf.py -m grid_sample -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float16 --dtypes float32 --warmup 50 --iter 100
```

**预期**: 需要实际测试结果

---

### 5. svd

**分支**: `codex/svd`

**状态**: 需要运行基准测试

**测试命令**:
```bash
pytest benchmark/test_special_perf.py -m svd -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float32 --warmup 50 --iter 100
```

**预期**: 需要实际测试结果

---

### 6. ctc_loss

**分支**: `codex/ctc_loss`

**状态**: 需要运行基准测试

**测试命令**:
```bash
pytest benchmark/test_reduction_perf.py -m ctc_loss -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float32 --warmup 50 --iter 100
```

**预期**: 需要实际测试结果

---

## 测试扩展结果

### 准确性测试总结

所有操作均已通过准确性测试：

1. **cosh**: ✅ 所有测试通过
2. **max_pool3d**: ✅ 24个测试通过 (前向12个 + 反向12个)
3. **avg_pool3d**: ✅ 48个测试通过 (前向24个 + 反向24个)
4. **grid_sample**: ✅ 18个测试通过 (2D: 9个 + 3D: 9个)
5. **svd**: ✅ 24个测试通过
6. **ctc_loss**: ✅ 所有测试通过

---

## 下一步

1. 为剩余操作运行性能基准测试
2. 收集加速比数据
3. 更新 PR 中的性能对比表
