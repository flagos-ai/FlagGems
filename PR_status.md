# Feb 27 KernelGen PR Status

## 已处理的 PR

| PR | 算子 | 状态 | 备注 |
|-----|------|------|------|
| [#1695](https://github.com/flagos-ai/FlagGems/pull/1695) | __rshift__ | ⏳ 待处理 | 超时，但是没问题 |
| [#1698](https://github.com/flagos-ai/FlagGems/pull/1698) | _convolution | ✅ 已修复 | 443 超时|
| [#1707](https://github.com/flagos-ai/FlagGems/pull/1707170) | _index_put_impl_ | ✅ 已修复 | rebase完成，push到colleague |
| [#1709](https://github.com/flagos-ai/FlagGems/pull/1709) | _scaled_dot_product_attention_math | ✅ 已修复 | rebase完成，push到colleague |
| [#1719](https://github.com/flagos-ai/FlagGems/pull/1719) | aminmax | 🔄 upstream已有 | 已在主仓库先不管了 |
| [#1720](https://github.com/flagos-ai/FlagGems/pull/1720) | argsort | ✅ 已修复 | test_pad upstream bug，加速比极低 |
| [#1722](https://github.com/flagos-ai/FlagGems/pull/1722) | bincount | ✅ 已修复 | rebase完成，大量test/benchmark冲突已解决，push到colleague |
| [#1723](https://github.com/flagos-ai/FlagGems/pull/1723) | clip | ✅ 已修复 | 加速比0.96，等待approve merge |
| [#1724](https://github.com/flagos-ai/FlagGems/pull/1724) | col2im | ✅ 已修复 | test_pad upstream bug |
| [#1725](https://github.com/flagos-ai/FlagGems/pull/1725) | concatenate | ✅ 已修复 | test_pad upstream bug |
| [#1726](https://github.com/flagos-ai/FlagGems/pull/1726) | conv_transpose1d | ✅ 已修复 | 依赖于conv1d，但是本身实现就有bug |
| [#1729](https://github.com/flagos-ai/FlagGems/pull/1729) | cudnn_convolution | ✅ 已修复 | 没benchmark 在找陶老师要 |
| [#1730](https://github.com/flagos-ai/FlagGems/pull/1730) | diff | ✅ 已修复 | pad 报错 |
| [#1731](https://github.com/flagos-ai/FlagGems/pull/1731) | einsum | ✅ 已修复 | 433报错|
| [#1732](https://github.com/flagos-ai/FlagGems/pull/1732) | feature_dropout | ✅ 已修复 | pad 报错 |
| [#1736](https://github.com/flagos-ai/FlagGems/pull/1736) | floor | 🔄 upstream已有 | upstream已有floor_先不提 |
| [#1737](https://github.com/flagos-ai/FlagGems/pull/1737) | fmod | ✅ 已修复 | 443超时报错 |
| [#1738](https://github.com/flagos-ai/FlagGems/pull/1738) | greater | 🔄 upstream已有 |先不提pr了 |
| [#1742](https://github.com/flagos-ai/FlagGems/pull/1742) | histc | ✅ 已修复 | pad 报错 |
| [#1743](https://github.com/flagos-ai/FlagGems/pull/1743) | index_copy_ | ✅ 已修复 | 等待 |
| [#1745](https://github.com/flagos-ai/FlagGems/pull/1745) | isneginf | 🔄 upstream已有 | |
| [#1747](https://github.com/flagos-ai/FlagGems/pull/1747) | log1p | 🔄 upstream已有 | upstream已有log1p_先不提pr了 |
| [#1748](https://github.com/flagos-ai/FlagGems/pull/1748) | log_normal_ | ✅ 已修复 | 啥也没有 |
| [#1749](https://github.com/flagos-ai/FlagGems/pull/1749) | logsumexp | ✅ 已修复 | 待approve and merge |
| [#1753](https://github.com/flagos-ai/FlagGems/pull/1753) | new_full | 🔄 upstream已有 | 先不提pr了|
| [#1754](https://github.com/flagos-ai/FlagGems/pull/1754) | nll_loss_nd | 🔄 upstream已有 |先不提pr了 |
| [#1755](https://github.com/flagos-ai/FlagGems/pull/1755) | nonzero_numpy | ✅ 已修复 | unittest 443 网络问题 |
| [#1757](https://github.com/flagos-ai/FlagGems/pull/1757) | poisson | ✅ 已修复 | 待approvemerge |
| [#1759](https://github.com/flagos-ai/FlagGems/pull/1759) | roll | ✅ 已修复 | 重写Triton kernel，修复取模bug，pytest全过，大tensor加速比~1.76x，push到colleague |
| [#1760](https://github.com/flagos-ai/FlagGems/pull/1760) | round | 🔄 upstream已有 |先不提pr了 |
| [#1761](https://github.com/flagos-ai/FlagGems/pull/1761) | rsub | ✅ 已修复 | 等待CI |
| [#1762](https://github.com/flagos-ai/FlagGems/pull/1762) | scatter_reduce_ | ✅ 已修复 | 待approve merge |
| [#1763](https://github.com/flagos-ai/FlagGems/pull/1763) | signbit | 🔄 upstream已有 | 先不提pr了|

## 汇总

- **总计**: 39个 PR
- **已合并**: 4个 (#1694, #1721, #1766, #1774)
- **upstream已有**: 9个 (#1719, #1736, #1738, #1745, #1747, #1753, #1754, #1760, #1763)
- **待merge**: 26个
- **待处理**: 0个

## 说明

所有需要修复的PR已完成：
- CLA作者修正为 factnn <1050552884@qq.com>
- 清理多余文件，只保留算子相关的5个核心文件
- 通过pre-commit检查
- Benchmark结果已发布到PR comment
