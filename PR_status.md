# Feb 27 KernelGen PR Status

## ✅ 可以 Approve Merge

| # | PR | 算子 | CI 状态 | 加速比 | Benchmark | 备注 |
|---|-----|------|---------|--------|-----------|------|
| 2 | [#1749](https://github.com/flagos-ai/FlagGems/pull/1749) | logsumexp | ✅ GREEN | **2.5-6.7x** | ✅ 已贴 | PR有问题：大量删除已注册算子 |
| 3 | [#1757](https://github.com/flagos-ai/FlagGems/pull/1757) | poisson | ✅ GREEN | 未测 | ❌ | 没有 rebase 到 master |
| 4 | [#1725](https://github.com/flagos-ai/FlagGems/pull/1725) | concatenate | ⏳ CLA | **1.1-1.4x** | ✅ 已贴 | __init__.py 合并冲突及小改动意见 |
| 5 | [#1729](https://github.com/flagos-ai/FlagGems/pull/1729) | cudnn_convolution | ⏳ CLA | 未测 | ✅ 已加 | ✅ Approved |
| 6 | [#1731](https://github.com/flagos-ai/FlagGems/pull/1731) | einsum | ⏳ CLA | ~0.7-0.9x | ✅ 已贴 | 已review，有疑问待回复 |
| 7 | [#1732](https://github.com/flagos-ai/FlagGems/pull/1732) | feature_dropout | ⏳ CLA | 0.3-0.8x | ✅ 已贴 | 已review，有疑问待回复 |
| 8 | [#1737](https://github.com/flagos-ai/FlagGems/pull/1737) | fmod | ⏳ CLA | ~1.0x | ✅ 已贴 | benchmark不全，算子名称有问题 |
| 9 | [#1743](https://github.com/flagos-ai/FlagGems/pull/1743) | index_copy_ | ⏳ CLA | 小tensor差, 大tensor~1.0x | ✅ 已贴 | 已review，小问题待修 |
| 10 | [#1755](https://github.com/flagos-ai/FlagGems/pull/1755) | nonzero_numpy | ⏳ CLA | 0.2-0.9x | 已跑未贴 | ✅ Approved |
| 11 | [#1761](https://github.com/flagos-ai/FlagGems/pull/1761) | rsub | ⏳ CLA | 未测 | ✅ 已加 | 有冲突待解决，有小修改建议 |
| 12 | [#1762](https://github.com/flagos-ai/FlagGems/pull/1762) | scatter_reduce_ | ⏳ CLA | 未测 | ✅ 已加 | 须细查现成实现，reduce支持多种方式 |
| 13 | [#1722](https://github.com/flagos-ai/FlagGems/pull/1722) | bincount | ⚠️ python-op | 未测 | ❌ | 需更细粒度精度/性能比较 |

## ⚠️ 重复算子（upstream已有实现，需更细粒度比较）

| # | PR | 算子 | CI 状态 | 备注 |
|---|-----|------|---------|------|
| 1 | [#1738](https://github.com/flagos-ai/FlagGems/pull/1738) | greater | ✅ GREEN | upstream已有，需更细粒度精度/性能比较 |
| 2 | [#1745](https://github.com/flagos-ai/FlagGems/pull/1745) | isneginf | ✅ GREEN | upstream已有，需更细粒度精度/性能比较 |
| 3 | [#1753](https://github.com/flagos-ai/FlagGems/pull/1753) | new_full | ✅ GREEN | upstream已有，需更细粒度精度/性能比较 |
| 4 | [#1754](https://github.com/flagos-ai/FlagGems/pull/1754) | nll_loss_nd | ✅ GREEN | upstream已有，需更细粒度精度/性能比较 |
| 5 | [#1763](https://github.com/flagos-ai/FlagGems/pull/1763) | signbit | ✅ GREEN | upstream已有，需更细粒度精度/性能比较 |

## ❌ 有问题，不应 Merge

| # | PR | 算子 | 问题 |
|---|-----|------|------|
| 1 | [#1695](https://github.com/flagos-ai/FlagGems/pull/1695) | __rshift__ | 等待unittest |
| 2 | [#1698](https://github.com/flagos-ai/FlagGems/pull/1698) | _convolution | hack无jit kernel |
| 3 | [#1709](https://github.com/flagos-ai/FlagGems/pull/1709) | _scaled_dot_product_attention_math | hack无jit kernel |
| 4 | [#1719](https://github.com/flagos-ai/FlagGems/pull/1719) | aminmax | upstream已有，跳过 |
| 5 | [#1720](https://github.com/flagos-ai/FlagGems/pull/1720) | argsort | 加速比极低(0.02x) |
| 6 | [#1724](https://github.com/flagos-ai/FlagGems/pull/1724) | col2im | test_pad upstream bug |
| 7 | [#1726](https://github.com/flagos-ai/FlagGems/pull/1726) | conv_transpose1d | kernel本身有bug |
| 8 | [#1730](https://github.com/flagos-ai/FlagGems/pull/1730) | diff | pad报错，1D极差 |
| 9 | [#1736](https://github.com/flagos-ai/FlagGems/pull/1736) | floor | upstream已有floor_ |
| 10 | [#1742](https://github.com/flagos-ai/FlagGems/pull/1742) | histc | 未测 |
| 11 | [#1747](https://github.com/flagos-ai/FlagGems/pull/1747) | log1p | upstream已有log1p_ |
| 12 | [#1748](https://github.com/flagos-ai/FlagGems/pull/1748) | log_normal_ | 无jit kernel，非fused实现 |
