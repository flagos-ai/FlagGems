# Feb 27 KernelGen PR Status (37个)

| # | PR | 算子 | CI 状态 | 加速比 | Benchmark | 备注 |
|---|-----|------|---------|--------|-----------|------|
| 1 | [#1694](https://github.com/flagos-ai/FlagGems/pull/1694) | __or__ | ✅ GREEN | ~1.0x | ✅ 已贴 | |   comment解决都等待approve merge
| 2 | [#1695](https://github.com/flagos-ai/FlagGems/pull/1695) | __rshift__ | ⏳ alpha-ops排队 | 已贴 | ✅ 已贴 | | 等待unittest
| 4 | [#1698](https://github.com/flagos-ai/FlagGems/pull/1698) | _convolution | ✅ GREEN | 未测 | ❌ | 无benchmark文件 |  monkeypatch done 等待approve merge
| 5 | [#1707](https://github.com/flagos-ai/FlagGems/pull/1707) | _index_put_impl_ | ✅ GREEN | ~1.0-1.5x | ✅ 已贴 | | 已approve待merge
| 7 | [#1709](https://github.com/flagos-ai/FlagGems/pull/1709) | _scaled_dot_product_attention_math | ✅ GREEN | ~1.0x | ✅ 已贴 | |  monkeypatch done 等待approve merge
| 8 | [#1719](https://github.com/flagos-ai/FlagGems/pull/1719) | aminmax | ❌ quick-cpu-op | - | - | upstream已有，跳过 | 已在主仓库
| 9 | [#1720](https://github.com/flagos-ai/FlagGems/pull/1720) | argsort | ⚠️ python-op | 0.02-2.4x | ✅ 已贴 | test_pad upstream bug |  FAILED tests/test_special_ops.py::test_pad[True-constant-dtype0-shape0] - RecursionError: maximum recursion depth exceeded while getting the repr of an object master 的 pad.py 里 constant_pad_nd 调用 pad，形成了无限递归，导致CI报错。加速比极低，不应该提到主仓库
| 10 | [#1721](https://github.com/flagos-ai/FlagGems/pull/1721) | bernoulli_ | ✅ GREEN | **1.1-1.2x** | ✅ 已贴 | |  等待approve merge
| 11 | [#1722](https://github.com/flagos-ai/FlagGems/pull/1722) | bincount | ⚠️ python-op | 未测 | ❌ | test_pad upstream bug |  test_pad upstream bug 
| 12 | [#1723](https://github.com/flagos-ai/FlagGems/pull/1723) | clip | ✅ GREEN | ~0.96x | ✅ 已贴 | | 加速比0.96，没有CI问题， 等待approve merge
| 13 | [#1724](https://github.com/flagos-ai/FlagGems/pull/1724) | col2im | ⚠️ python-op | 未测 | ❌ | | test_pad upstream bug 
| 14 | [#1725](https://github.com/flagos-ai/FlagGems/pull/1725) | concatenate | ⏳ CLA | **1.1-1.4x** | ✅ 已贴 | |  test_pad upstream bug 
| 15 | [#1726](https://github.com/flagos-ai/FlagGems/pull/1726) | conv_transpose1d | ⏳ CLA | fp16: **1.2-3.3x**, fp32: 0.5-1.1x | ✅ 已贴 |    依赖于conv1d，但是本身实现就有bug
| 16 | [#1729](https://github.com/flagos-ai/FlagGems/pull/1729) | cudnn_convolution | ⏳ CLA | 未测 | ❌ | 无benchmark文件 | 没benchmark 在找陶老师要
| 17 | [#1730](https://github.com/flagos-ai/FlagGems/pull/1730) | diff | ⏳ CLA | 多维1.2x, 1D极差 | ✅ 已贴 | | pad 报错
| 18 | [#1731](https://github.com/flagos-ai/FlagGems/pull/1731) | einsum | ⏳ CLA | ~0.7-0.9x | ✅ 已贴 | |
| 19 | [#1732](https://github.com/flagos-ai/FlagGems/pull/1732) | feature_dropout | ⏳ CLA | 0.3-0.8x | ✅ 已贴 | |

| 20 | [#1736](https://github.com/flagos-ai/FlagGems/pull/1736) | floor | ⏳ CLA | 冲突 | ❌ | upstream已有floor_ |
| 21 | [#1737](https://github.com/flagos-ai/FlagGems/pull/1737) | fmod | ⏳ CLA | ~1.0x | ✅ 已贴 | |
| 22 | [#1738](https://github.com/flagos-ai/FlagGems/pull/1738) | greater | ✅ GREEN | - | ❌ | upstream已有 |
| 23 | [#1742](https://github.com/flagos-ai/FlagGems/pull/1742) | histc | ⏳ CLA | 未测 | ❌ | benchmark语法错误 |
| 24 | [#1743](https://github.com/flagos-ai/FlagGems/pull/1743) | index_copy_ | ⏳ CLA | 小tensor差, 大tensor~1.0x | ✅ 已贴 | |
| 25 | [#1745](https://github.com/flagos-ai/FlagGems/pull/1745) | isneginf | ✅ GREEN | - | ❌ | upstream已有 |
| 26 | [#1747](https://github.com/flagos-ai/FlagGems/pull/1747) | log1p | ⏳ CLA | 冲突 | ❌ | upstream已有log1p_ |
| 27 | [#1748](https://github.com/flagos-ai/FlagGems/pull/1748) | log_normal_ | ⏳ CLA | 未测 | ❌ | diff脏(30+文件) |
| 28 | [#1749](https://github.com/flagos-ai/FlagGems/pull/1749) | logsumexp | ✅ GREEN | **2.5-6.7x** | ✅ 已贴 | 最佳 |
| 29 | [#1753](https://github.com/flagos-ai/FlagGems/pull/1753) | new_full | ✅ GREEN | - | ❌ | upstream已有 |
| 30 | [#1754](https://github.com/flagos-ai/FlagGems/pull/1754) | nll_loss_nd | ✅ GREEN | - | ❌ | upstream已有 |
| 31 | [#1755](https://github.com/flagos-ai/FlagGems/pull/1755) | nonzero_numpy | ⏳ CLA | 0.2-0.9x | 已跑未贴 | |
| 32 | [#1757](https://github.com/flagos-ai/FlagGems/pull/1757) | poisson | ✅ GREEN | 未测 | ❌ | |
| 33 | [#1759](https://github.com/flagos-ai/FlagGems/pull/1759) | roll | ⏳ CLA | 未测 | ❌ | diff脏(20+文件) |
| 34 | [#1760](https://github.com/flagos-ai/FlagGems/pull/1760) | round | - | - | - | upstream已有，跳过 |
| 35 | [#1761](https://github.com/flagos-ai/FlagGems/pull/1761) | rsub | ⏳ CLA | 未测 | ❌ | diff脏(多余文件) |
| 36 | [#1762](https://github.com/flagos-ai/FlagGems/pull/1762) | scatter_reduce_ | ⏳ CLA | 未测 | ❌ | diff为空 |
| 37 | [#1763](https://github.com/flagos-ai/FlagGems/pull/1763) | signbit | ✅ GREEN | - | ❌ | upstream已有 |
| 38 | [#1766](https://github.com/flagos-ai/FlagGems/pull/1766) | unique_consecutive | ⏳ CLA | 未测 | ❌ | |
| 39 | [#1774](https://github.com/flagos-ai/FlagGems/pull/1774) | var | ⏳ CLA | 未测 | ❌ | diff脏(多余workflow文件) |

## 汇总
- **已合并**: 2个 (#1697, #1708)
- **CI 全绿**: 12个
- **CLA pending**: 17个（需要签 CLA）
- **python-op 失败**: 3个（都是 upstream test_pad bug）
- **upstream 已有可跳过**: 9个 (#1719, #1736, #1738, #1745, #1747, #1753, #1754, #1760, #1763)
- **Diff 脏需清理**: 4个 (#1748, #1759, #1761, #1774)
- **Benchmark 已贴**: 12个
- **加速比最好**: #1749 logsumexp (2.5-6.7x)
