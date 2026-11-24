# Ascend TensorView 算子集成报告

本文记录把 FlagTree TensorView 原语（`tle.make_partition_view` /
`make_strided_view` / `make_gather_scatter_view` + `tile_load` / `tile_store`）
接入 FlagGems Ascend 后端的集成方式、回退设计、功能测试结果与性能数据，并逐条
列出失败用例及其报错。

实现入口是 `tensorview.py`（可用性闸门与 UB 预算助手），各算子在
`ops/` 与 `fused/` 下以 `SpecOpRegistrar` 覆盖通用实现。

---

## 0. 第二轮：对齐 `bounds_check` API 并补齐到 17 个算子

本节是 2026-11-17 这一轮的改动。**第 1 节起的内容描述的是第一轮**（11 个算子、
旧 API），其中的测试与性能数据未在本轮环境下复测，仍按当时结论保留。

### 0.1 上游 API 变化：`bounds_check`

FlagTree 当前分支（`triton_v3.5.x` + 未合入的 merge `f7ca62bad
[TLE][CommonIR] Add TensorView operator examples`）给
`make_gather_scatter_view` 加了 `bounds_check` 参数：

- `bounds_check=True`：越界的 sparse 下标在 load 时读成 `padding_value`、在 store
  时被跳过。于是 `-1` 成为合法的「跳过该行」标记，尾块不需要显式 mask，且
  `shape` 的每一维都必须给出。
- `bounds_check=False`（默认）：sparse 下标必须落在 `[0, shape[d])` 内，**越界是
  未定义行为**；sparse 维的 `shape[d]` 可以是 `None`，不进 kernel 参数。

第一轮写在旧 API 上，没有这个参数却已经在传 `-1`，也就是一直踩在未定义行为上。
本轮把除 `paged_kv_block_gather` 以外的所有 GSV 调用都改成 `bounds_check=True`
（该算子的下标来自 block table，保证有效，新示例同样没开）。

### 0.2 `cp_gather_indexer_k_quant_cache` 重新启用

第一轮因为 device 报 `vector core exception` 把它关掉了（见第 6.5 节）。原因基本
可以确定就是 0.1 里那条未定义行为：无效 token 的行号取 `-1`，而视图没开边界检查。
本轮改成 `bounds_check=True` 并删掉 `_TV_ENABLED = False`。

**这一条没有在设备上复测过**——本机安装的 flagtree 没有 TensorView API，所有调用
都回退到通用实现。上快路径环境后需要重新验证。

其 eligibility 依然很窄，但原因与上述故障无关：每个 block 行把 fp32 scale 内联在
值之后，只有当这个带 padding 的行跨距正好是 `HEAD_DIM` 的整数倍时，行才能按 tile
下标寻址。上游 10 个测试形状里只有 1 个满足（`block_size=64, head_dim=512`），
其余 9 个走通用实现。

### 0.3 `MIN_TILE_ROWS = 2`

新示例有三处明确写了「A one-row tile does not compile, so the tile height is at
least 2」。`tile_rows_for_ub` 因此加了 `min_rows` 参数，默认仍是 1。

注意**不能**直接拿钳位后的结果做 eligibility 判断：宽行上钳位会把 1 抬成 2，而这
2 行本来就装不进 UB，照此启动会编译失败。所以本轮新增的算子都分成两个函数——
`_tv_rows_that_fit()` 返回未钳位的真实容量（给闸门用），`_tv_block_m()` /
`_tv_block_t()` 返回钳位后的启动值。闸门判 `_tv_rows_that_fit(...) >=
MIN_TILE_ROWS`。

第一轮两个 norm 用的是 `min_rows=1` 加 `> 1` 判断，本身就等价，未改动。

### 0.4 两个 norm 的输出视图改回 PartitionView

新示例里 `rms_norm` 与 `fused_add_rms_norm` 的输出不再走 GSV + 显式行号，而是用
PartitionView 按 tile 下标寻址（`index=(row_block, 0)`）。`fused_add_rms_norm`
进一步直接复用读入时的那两个视图写回，省掉两个输出视图和行号的构造。

### 0.5 本轮新增的 6 个算子

| 算子 | 文件 | 视图 | 闸门要点 |
|---|---|---|---|
| skip_layer_norm | `fused/skip_layernorm.py` | PartitionView ×5 | `len(normalized_shape)==1`、宽度为 2 的幂、UB 能装下 ≥2 行 |
| apply_rotary_pos_emb | `fused/rotary_embedding.py` | 3-D PartitionView ×4 | 仅 half-split（`rotary_interleaved=False`）、非 inplace、须给 `position_ids`、`head_dim//2` 为 2 的幂（故 `head_dim=96` 走通用） |
| pack_seq_triton | `fused/pack_seq.py`（新增） | StridedView ×2 | `lengths` 须在 device 上、序列数 ≤1024（前缀扫描是每 program 一遍） |
| unpack_seq_triton | `fused/unpack_seq.py`（新增） | StridedView ×2 | 同上；输出视图止于序列最后一行，尾行由视图裁掉 |
| combine_topk_swa_indices | `fused/deepseek_v4_attention_combine_topk_swa_indices.py`（新增） | StridedView ×3 | `compress_ratio > 0`（kernel 要除它）、`window_size > 0`、top-k 行须紧致 |
| compute_global_topk_indices_and_lens | `fused/deepseek_v4_attention_compute_global_topk_indices_and_lens.py`（新增） | PartitionView ×5 | `block_size > 0`、输入 int32、token 侧张量紧致 |

`pack_seq` / `unpack_seq` 的示例有一招值得单独记：起点落在序列末尾之后的 tile 会
被拉回序列首行再 load，load 完再按 `valid` 掩掉。这样每次 load 都走整 tile DMA，
而不是退化成逐元素填充。

`apply_rotary_pos_emb` 的 FlagGems 签名比示例宽（多了 `rotary_interleaved`、
`inplace`、`position_ids=None`、任意前导维），闸门按上表收窄；`position_ids=None`
时位置是每个 batch 隐含的 `0..seq_len-1`，要在 host 侧物化出来才能喂给 kernel，
而 host 函数不允许调 PyTorch 计算算子，所以这一路直接留给通用实现。

### 0.6 顺手修掉的一处注释与实现不符

`src/flag_gems/fused/paged_kv_block_gather.py`（第一轮随快路径一起新增的公共实现）
的 kernel 注释写着越界 block id「leave the destination zeroed」，但 host 侧是
`torch.empty` 分配，掩掉的 store 根本不会写，那个槽位留的是未初始化内存。写验证
脚本时按注释断言，于是测出来不符。

**没有改成 `torch.zeros`**：那会给每次调用都加一遍 memset，而越界 block id 在真实
paged cache 里不会出现；快路径那边的 GSV 也没开 `bounds_check`（下标由 block table
保证有效），改了也对不齐。所以改的是注释，把实际行为写清楚：越界槽位内容未定义，
调用方不能依赖；同时在 `_ascend` 快路径里把「这个视图故意不开 bounds_check、没有
-1 跳过语义」也写进注释，和其他几个视图区分开。

### 0.7 `index_put` 闸门收窄：去掉 host 侧的 `torch.broadcast_to`

`tools/ci_checks/check_no_torch_fallback.py --base ttt` 报出第一轮留下的一处违规：

```
src/flag_gems/runtime/backend/_ascend/ops/index_put.py:150:
    forbidden torch compute call torch.broadcast_to()
```

第一轮的 `_launch_index_put_tv` 写的是
`torch.broadcast_to(values, (num_indices, *out.shape[1:])).contiguous()`，用来容纳
FlagGems 比示例更宽的签名（`values` 可以只有一行、靠广播铺到所有下标）。
`broadcast_to` 本身只是造个 stride 为 0 的视图，但紧跟的 `.contiguous()` 会真的
复制一遍 —— 这就是 AGENTS.md 禁止的「host 函数调 PyTorch 计算算子」。

示例 `index_put.py` 的契约里 `values` 本来就是 `[num_indices, width]`，没有广播这回事；
kernel 通过 `[num_indices, WIDTH]` 的 PartitionView 按 tile 下标读它，也要求内存里
每个下标都有真实的一行。所以这里**按示例的契约收窄闸门**，而不是加
`# allow-torch-compute` 把检查绕过去：

```python
if values.numel() != index.numel() * math.prod(row_shape):
    return False
```

需要广播的 `values`（`[1, W]`、裸 `[W]`）从此判不合格、走通用实现 —— 通用实现本来
就处理这些形状。快路径少覆盖了一类输入，换来的是 host 函数里不再有 PyTorch 复制。
新增 9 条断言覆盖这条规则的两侧（含 3 维目标张量的 `row_shape` 乘积情形）。

### 0.8 验证范围

本机安装的是 flagtree `0.7.0+ascend.gitdf9531a7`，**完全没有 TensorView API**：

```
AttributeError: module 'triton.experimental.tle' has no attribute 'make_gather_scatter_view'
```

因此 `HAS_TENSORVIEW` 为 False，快路径 kernel 根本不会定义。本轮能验证的是：

1. **覆盖注册**：用 stub 把 `tle` 命名空间假装齐全、强制 `HAS_TENSORVIEW=True`，
   确认 20 个入口（17 个算子，含 `index_add_` / `index_copy_` / `index_put_` 三个
   原地形态）全部解析到 `_ascend.*` 覆盖实现。
2. **eligibility 闸门**：闸门是纯 Python，可在本机直接跑。39 条断言（30 条覆盖
   上表每条收窄条件的接受/拒绝两侧，另 9 条是 0.7 节那条新规则）全部符合预期。
3. **回退路径无回归**：逐套跑 FlagGems 测试，与 `ttt` 基线比对失败集合。

第 3 项的结果：1564 个用例通过，失败集合与基线一致，本轮改动没有引入新失败。
唯一需要单独说明的是 `tests/test_index_add.py::test_index_add_empty_index` 的
8 个用例（4 个 dtype × `True`/`False`），报 `ZeroDivisionError`：

```
src/flag_gems/runtime/backend/_ascend/ops/index_add.py:316: in _index_add_generic
    M = src.numel() // N
E   ZeroDivisionError: integer division or modulo by zero
```

`N = index.numel()`，空下标时为 0。这是**既有问题**，不是本轮引入：同一行在 `ttt`
上位于 `index_add.py:305`，从 `git archive ttt` 解出干净副本单跑该用例，得到完全
相同的 `8 failed`。第三轮一并修掉了，见 0.11 节。

**没有验证的是快路径本身的数值与性能。** 需要先解掉 `new/FlagTree` 的
submodule `commit_id` merge 冲突、重新编译安装，才能复测第 5、7 节那些数据。

### 0.9 第三轮：装上带 TensorView 的 flagtree 之后

flagtree 重装为 `0.7.0+ascend.git92789138`（editable 指向 `new/FlagTree/python`），
五个符号齐全、签名与移植的调用逐项对得上，`HAS_TENSORVIEW` 终于为 True。这是
第一次能真正验证快路径，也立刻暴露了两个在 `HAS_TENSORVIEW=False` 下不可能发现
的问题。

先是一个移植自身的 bug：`_has_tile_transfer()` 和 17 个 kernel 文件都从
`tle.language.dsa` 导入 `tile_load` / `tile_store`，而这两个函数定义在
`tle/language/tile.py`、由 `tle` 本身导出，`dsa` 包导出的是
`tile_alloc` / `tile_copy` / `tile_to_tensor` 那一族。FlagTree 自己的 17 个示例
一律写 `tle.tile_load(...)`，没有一个走 dsa。探针把 ImportError 吞掉返回 False，
而这与「工具链没有 TensorView」的表现完全一致，所以错误一直被掩盖着：旧构建上
`HAS_TENSORVIEW=False` 是对的，新构建上也会是 False 但原因是错的——装上可用的
flagtree 后，所有快路径会静默地完全不生效。改为从 `tle` 导入后探针转 True，
17 个模块的 kernel 与闸门都正常定义（commit `df127cf3d`）。

### 0.10 七个闸门漏掉 UB 预算检查

`HAS_TENSORVIEW` 为 True 后跑 `tests/test_index_select.py`，18 个用例全过，但
闸门终态是 `_BAD`——有一次编译失败被 `tensorview_call` 吸收了。开日志级别抓到：

```
GEMS_ASCEND INDEX_SELECT: the TensorView path does not compile ...
[NOTE] Due to Ub overflow, automatically set enable-code-motion to false,
       set enable-auto-multi-buffer to false; compilation still failed.
error: Failed to run BiShengHIR pipeline
```

肇事形状是 `REDUCTION_SHAPES` 里的 `(200, 40999, 3)`：尾部维度折叠成一行是
40999×3 = 122997 元素，向上取整到 131072 宽的 tile，fp32 下单行 512 KB，而
`_UB_BUDGET` 只有 96 KB——连一行都放不下。

`tile_rows_for_ub` 的下界是 1，并在 docstring 里把「单行放不下」明确留给调用方的
eligibility 检查（见 0.3 节的设计意图）。问题是 A 组里有七个算子只在 launch 路径
算 tile 高度，闸门里从没做这个检查：`index_select`、`embedding`、`index_add`、
`index_copy_`、`index_put`、`concat_and_cache_mla`、`reshape_and_cache_flash`。
B 组六个新算子全都按约定写了 `_tv_rows_that_fit(...) >= MIN_TILE_ROWS`，两个 norm
也有等价的 `_tv_block_m(width) > 1`，漏的恰好是本轮改写的这七个。

修法是在 `tensorview.py` 新增 `tile_fits_ub(row_bytes, live_tiles)`，回答
「MIN_TILE_ROWS 行能否放进 UB」。`tile_rows_for_ub` 把答案压到 1，所以分不出
「一行刚好放得下」和「一行都放不下」，而这两种都会发射溢出 UB 的 tile。七处闸门
各加一行，**参数与各自 launch 路径传给 `tile_rows_for_ub` 的完全一致**
（`live_tiles` 各不相同：`index_add` 是 2、`index_put` 看 `accumulate`、
`reshape_and_cache_flash` 是 2、其余为 1），避免两处算术再次漂移——这次的 bug 正是
这么来的。14 条断言（每个算子一个正常形状 + 一个超宽形状）确认闸门按预期拒绝，
正常形状未被误杀。

### 0.11 `index_add` 空下标除零

0.8 节记录的既有问题，本轮一并修掉。测试本身写明了期望语义：

```python
def test_index_add_empty_index(inplace, dtype):
    """An empty index is a no-op, not a division by zero (issue #6657)."""
```

在 `N = index.numel()` 之后、除法之前拦掉。两种形态的返回值不同，不能共用：
`_index_add_generic` 返回 `inp.clone()`（异地形式须返回新张量且不得改动输入，
测试第 174 行专门检查了这条），`_index_add__generic` 返回 `inp` 本身。
`tests/test_index_add.py` 从 `56 passed / 8 failed` 转为 **64 passed**。

这一项与 TensorView 无关，属于通用实现里的既有缺陷。

### 0.12 快路径的数值与性能（首次实测）

CANN 9.1.0 的 `bishengir-opt` 1.2.0 读不了 flagtree 产出的 MLIR bytecode v6
（`expected mlir::hivm::TCoreTypeAttr`），而 v5 起 MLIR 会写 per-dialect version
段、要求读写两侧方言版本一致。把容器格式钉到 v4 即可绕开，产出的 IR 与匹配版本
reader 从 v6 解出的逐行相同。这是 FlagTree 侧的适配（`backends/ascend/compiler.py`
探测后给 `triton-mlir-opt` 加 `--emit-bytecode-version`），不在本仓改动范围内，
此处仅记录依赖条件。

精度：16 套逐套独立进程跑完，失败集合与基线一致，无新增失败，无任何算子降级。
`gate_state` 为 `GOOD`（仅在快路径成功执行后置上）的八个算子确认快路径真实生效。

性能：同一输入分别走 `_<op>_tensorview` 与 `_<op>_generic`，20 次取中位数，
并校验两条路数值一致。14 个用例全部数值一致，12 个快于通用，中位 4.72x：

| 算子 | 形状 | 通用 | TensorView | 加速 |
|---|---|---|---|---|
| reshape_and_cache_flash | tokens=4096 heads=8 dim=128 | 27.250 ms | 0.284 ms | 96.06x |
| index_select | rows=32000 width=512 idx=4096 | 14.467 ms | 0.221 ms | 65.33x |
| index_put | rows=16384 width=512 idx=4096 | 11.711 ms | 0.257 ms | 45.55x |
| index_add | rows=16384 width=512 idx=4096 | 11.632 ms | 0.297 ms | 39.20x |
| reshape_and_cache_flash | tokens=512 | 3.676 ms | 0.275 ms | 13.37x |
| index_select | rows=4096 width=256 idx=512 | 1.388 ms | 0.209 ms | 6.64x |
| index_put | rows=4096 width=256 idx=512 | 1.263 ms | 0.254 ms | 4.97x |
| index_add | rows=4096 width=256 idx=512 | 1.160 ms | 0.259 ms | 4.48x |
| index_copy | rows=16384 width=512 idx=4096 | 0.771 ms | 0.255 ms | 3.02x |
| rms_norm | rows=4096 width=1024 | 0.426 ms | 0.232 ms | 1.84x |
| rms_norm | rows=16384 width=4096 | 1.345 ms | 0.764 ms | 1.76x |
| embedding | vocab=128000 dim=512 tokens=4096 | 0.246 ms | 0.209 ms | 1.18x |
| index_copy | rows=4096 width=256 idx=512 | 0.283 ms | 0.293 ms | 0.97x |
| embedding | vocab=32000 dim=256 tokens=512 | 0.161 ms | 0.204 ms | 0.79x |

最值得注意的规律：**快路径耗时几乎不随规模变化**。`index_select` 从 4096 行涨到
32000 行，通用实现 1.388 ms → 14.467 ms（约 10 倍），快路径 0.209 ms → 0.221 ms
基本不动。gather/scatter 搬运在 DMA 引擎上按 tile 流水，开销被摊掉；通用实现按
元素寻址，规模一上来就线性恶化。所以大形状上的 40~96 倍不是小形状那个倍数的延伸。

两个慢于通用的都是小形状，且差值都在 0.04 ms 以内：这台机器上 TensorView kernel
的地板约 0.2 ms（视图构造的固定开销），而这两个用例的通用实现本就只要
0.16~0.28 ms，已接近 kernel 启动开销。属于固定开销占比问题，不是实现缺陷。给这
两个算子的闸门加规模下界是可选的优化，但阈值随机器变化，当前未加——闸门每多一条
就多一处需要维护的分歧。

### 0.13 两个 sub-1x 的归因（三维扫描）

把 `index_copy` 和 `embedding` 的三个维度分别固定、单独扫描，确认 0.97x 与 0.77x
都是固定启动开销，不是搬运惩罚。

快路径成本几乎不随规模变，只有真正决定工作量的那一维能让通用实现爬出地板：

| 算子 | 不影响工作量的维度 | 决定工作量的维度 |
|---|---|---|
| `index_copy` | rows 512→32768，两条路径都稳在 0.267/0.275 ms，全程 0.97x | width 32→2048（0.347→0.494 vs 0.354→0.275，1.79x）；nidx 16→4096（1.78x） |
| `embedding` | vocab 4096→256000、dim 32→2048，全程 0.77x | tokens 16→8192（0.163→0.402 vs 0.215→0.216，1.86x） |

空 kernel 启动一次 0.114 ms，是本机测量地板。`embedding` 通用路径的 0.167 ms 里
0.054 是输出分配、约 0.114 是启动，本身没有可测的工作量；快路径多出的 0.048 ms
在基准形状和小 500 倍的形状（64×32、16 token）上分别是 0.049 与 0.047 ms，跨这个
规模比值不变，证明它是每次调用的固定成本。`index_copy` 的差值约 6 µs（占 2~3%），
原地形式直接打平（0.176 vs 0.177 ms）。`embedding` 的交叉点压到 tokens≈3072。

顺带发现一处通用实现的硬限制：`embedding` 一个 token 一个 program，tokens 超过
65535 时撞 Ascend coreDim 上限直接起不来

    Invalid_Argument(EE1003): value 65536 for parameter coreDim is invalid.
    Expected value: less than or equal to 65535.

快路径按 tile 切，grid 是 tokens/BLOCK_M，实测 65536 与 131072 都正常且结果正确
（对 CPU gather 校验）。上游在本轮给 `rms_norm` 加了 `MAX_CORE_DIM` 封顶，
`embedding` 这条还没修。这也是不给 `embedding` 闸门加规模下界的理由：小形状亏
48 µs，而大 token 数下快路径是唯一能跑的。

### 0.14 第四轮：迁到上游重组后的新布局

上游 `9c36600eb` 把 vLLM 系算子整体搬了位置，旧布局作废，于是在 `tv_new` 分支上
按新布局重做了一遍移植。三处结构性变化：

- **`_ascend/fused/` 整个目录不再存在。** 后端覆盖现在只扫 `_ascend/ops` 与
  `_ascend/deprecated`（`backend/__init__.py:298` 把 `fused_module` 指向
  `_{vendor}.deprecated`），所以原来 `fused/` 下的 11 个覆盖文件全部落到
  `_ascend/deprecated/`。
- **通用实现分散到三处**，覆盖文件的 fallback import 要分别对接：
  `concat_and_cache_mla`、`reshape_and_cache_flash` 仍在 `flag_gems.fused`；
  `cp_gather_indexer_k_quant_cache`、两个 deepseek、`pack_seq`、`unpack_seq` 搬到
  `flag_gems.deprecated`；`paged_kv_block_gather` 是本次新增的公共算子，仍在
  `flag_gems.fused`。
- **`rms_norm` 的闸门位置变了。** 上游把 `rms_norm_forward` 提成模块级函数并注明
  TransformerEngine 直接调它，所以闸门放进 `rms_norm_forward` 而不是
  `RmsNorm.forward`，否则 TE 那条路拿不到快路径。该函数同时多了 `M == 0 or N == 0`
  的早退和 `min(M, MAX_CORE_DIM)` 封顶，快路径在其后插入。

上游本轮自己修掉的，本次不再重复移植：`index_add` 空下标除零（写法与 0.11 节一
字不差）、host 侧 `aminmax` 换成 device 端 `_index_add_bounds_kernel`（顺带解决了
AGENTS.md「host 函数不得调 PyTorch 计算算子」那条）、`MAX_CORE_DIM` 封顶。

注册侧补了两处原本漏掉的导出：`_ascend/ops/__init__.py` 的 `index_put` / `index_put_`
（不补则这两个入口根本不进 dispatcher），以及 `conf/operators.yaml` 的
`paged_kv_block_gather` 条目。

### 0.15 上游新测试抓出的三个真 bug

上游本轮给 `index_add` 和 `rms_norm` 加了新用例，基线 170 / 27 全过，装上快路径后
变成 20 / 3 失败——**全部是快路径自己的问题**，不是测试与实现的口径差异。逐个查清
如下。

**（1）读-改-写丢更新：行小于一个写粒度时，相邻目标行互相覆盖。**

最严重的一个。`index_add` 的快路径是在同一个 GatherScatterView 上做
read-modify-write，`tests/test_index_add.py::test_index_add_repeated_indices`
报出目标行 2 的更新整个丢了（得 2.0，应 -4.0）。

逐步缩小后规则很干净，与下标是否重复无关，只跟**行字节**和**目标行间距**有关：

| 行字节 | 相邻下标（gap=1~3） | gap≥4 |
|---|---|---|
| 4 B（fp32 宽 1） | 全丢 | 全丢 |
| 8 / 16 B | 丢 | 正常 |
| 24 B 及以上 | 正常 | 正常 |

fp16 宽 8（16 B）丢、宽 12（24 B）不丢；fp32 宽 4（16 B）丢、宽 8（32 B）不丢——
阈值按**字节**算，与 dtype 无关，正是一个 DMA 写粒度的尺寸。行比粒度窄时两个目标
行共用一个粒度，read-modify-write 就会丢掉其中一个。`num_rows` 完全无关（32~4096
表现一致），换成两个独立的 view 对象也没用，改成 `tl.atomic_add` 则全对。

纯 scatter 不受影响：`index_put(accumulate=False)`、`index_copy` 在 4 B 行上也全对。
所以这是 read-modify-write 专有的约束。

修法是在两个 RMW 闸门上加 `row_bytes >= 32`（`_SCATTER_GRANULE_BYTES`）：
`index_add` 和 `index_put(accumulate=True)`。闸门不能依赖下标值，所以只能按粒度
一刀切，窄行交给通用实现——通用实现用的是另一套分解，没有这个约束。

`index_put` 的 accumulate 分支有同样的 bug 且闸门原本放行 16 B 行，是顺着
`index_add` 查出来的，上游测试没覆盖到：直接调快路径，`(64,4)` fp32（16 B 行）
与 `(64,8)` fp16（16 B 行）在相邻下标下都算错，`(64,8)` fp32（32 B 行）正确。
补上粒度闸门后这三个形状的判定变成 False / False / True。

`tests/test_index_put.py` 本身有 88 个既有失败，与本次改动无关：把整个
`_ascend/ops/index_put.py` 摘掉（退回公共实现）是 88 失败 63 通过，把闸门强制
返回 False 也是 88 / 63，失败集合逐条相同。所以这 88 条属于通用实现，TensorView
层没有引入新失败。

**（2）图捕获下的数据相关闸门。**

`test_index_add_graph_replay` 全部 12 例失败。`has_no_duplicates` 要
`.item()` 同步，捕获期不允许；更根本的是这个闸门**读下标值**，捕获时判定"无重复"
会被固化进图，而 replay 时下标缓冲区里换成了全 15（全重复），判定已经不成立。
通用 kernel 每次执行都对下标做 mask，replay 安全。所以两个 RMW 闸门都加了
`is_current_stream_capturing()` 早退，与上游 `_assert_index_in_bounds` 的做法一致。

**（3）越界下标必须报错，而 `bounds_check` 把它吞了。**

`test_index_add_short_invalid_preserves_input[-1]` 要求下标 -1 抛
`0 <= index < self.size(dim)` 并保持输入不变。快路径开着 `bounds_check=True`，
-1 被当成"跳过该行"静默忽略，异常不再抛出。该用例的输入是 1-D（行 4 B），加上
（1）的粒度闸门后已经走通用实现，断言恢复，无需额外改动。

**（4）快路径自己也会撞 coreDim 上限。**

`rms_norm` 的 3 例失败最初看着只是"测试断言通用 kernel 必须启动一次"的口径问题，
但顺着查发现快路径有真问题：`grid = cdiv(M, BLOCK_M)` 没有封顶，`M=2000000`、
`N=512` 时 grid=125000 > 65535，启动直接失败——正是上游本轮给通用路径修掉的那个
bug，快路径又引入了一遍。改成和通用一致的 `min(..., MAX_CORE_DIM)` 并让 kernel 按
`tl.num_programs(0)` 步长走完所有 tile，实测 M=65536 / 786432 / 2000000 都正确。
视图不依赖 tile 下标，提到循环外构造一次。

上游那个用例断言三个 generic kernel 各启动一次，快路径接管前向后
`rms_norm_kernel` 启动 0 次。把 TV kernel 也纳入记录、要求"前向恰由两者之一服务
一次"，粒度上限仍对实际启动的那个 kernel 校验——守护的意图没有削弱。

### 0.16 第五轮：越界断言的可达缺口，与闸门同步开销的真实代价

0.15（3）说"加上粒度闸门后那个用例已走通用，断言恢复"——对那个用例成立，它是 1-D
输入、行 4 B。但**这个结论不能推广**：换成闸门接受的宽度，缺口依然存在。

**（5）越界下标在快路径被静默跳过（真 bug，已修）。**

用 2-D fp32、宽 8（32 B，刚好一个写粒度，闸门接受）探测 `index_add`：

```
index=[0,-1]  gate accepts: True   -> 无异常返回，行 0 的更新照常落地
index=[0,64]  gate accepts: False  -> 被 has_no_duplicates 的 scatter_ 拒掉
```

`-1` 这一侧闸门放行，`bounds_check=True` 把它当"跳过该行"静默吞掉，而通用实现
`_assert_index_in_bounds` 会抛 `0 <= index < self.size(dim)`。上界那一侧只是**碰巧**
被拦住：`has_no_duplicates` 用 `scatter_(0, index.long(), ...)`，下标等于 `num_rows`
时 scatter 自己失败，于是闸门返回 False——不是有意的边界检查。

把断言移进 `_launch_index_add_tv`（两个入口都经由它），放在 launch 之前，于是原地
调用在报错时输入不变，与通用实现同契约、同消息。修完 `index=[0,-1]` 正常抛出，
`tests/test_index_add.py` 仍 170 过。

同样的探测跑了另外四个算子，结论是**只有 `index_add` 存在分歧**：

| 算子 | 通用路径（下标 -1 / 64） | 快路径 | 判定 |
|---|---|---|---|
| index_add | 抛 AssertionError | 静默跳过 | 分歧，已修 |
| index_copy | 静默返回 | 静默跳过 | 一致 |
| index_put | 静默返回 | 静默跳过 | 一致 |
| index_select | 静默返回 | 静默跳过 | 一致 |
| embedding | 静默返回 | 静默跳过 | 一致 |

torch 自己对这五个都抛异常，但快路径的契约是它覆盖的那个通用实现，不是 torch，
所以后四个无需改动。

**（6）0.12 的性能表测的不是生产路径（口径问题，非 bug）。**

`bench_tensorview.py` 直接比 `_<op>_tensorview` 与 `_<op>_generic`，按设计绕开了
eligibility 闸门。五个算子的闸门纯看形状，绕不绕开没差别；但 `index_add` 与
`index_put(accumulate=True)` 的闸门要调 `has_no_duplicates`，那是一次
device→host 同步。实测这次同步比它守护的 kernel 贵一个量级：

| | small | large |
|---|---|---|
| TV kernel 本身 | 0.138 ms | 0.148 ms |
| `has_no_duplicates`（闸门） | 0.392 ms | 0.566 ms |
| `_assert_index_in_bounds`（本轮新加） | 0.295 ms | 0.401 ms |
| **TV 合计** | **0.825 ms** | **1.115 ms** |
| 通用合计 | 0.588 ms | 0.659 ms |

于是改用生产入口（走闸门）重测全部算子，16 例数值全一致，11 例快于通用，中位 3.37x：

| 算子 | 形状 | 通用 | 带闸门 TV | 加速 |
|---|---|---|---|---|
| reshape_and_cache_flash | tokens=4096 heads=8 dim=128 | 27.308 ms | 0.300 ms | 90.89x |
| index_select | rows=32000 width=512 idx=4096 | 13.915 ms | 0.242 ms | 57.52x |
| index_put(accumulate) | rows=16384 width=512 idx=4096 | 43.920 ms | 0.825 ms | 53.26x |
| index_put | rows=16384 width=512 idx=4096 | 11.718 ms | 0.300 ms | 39.11x |
| reshape_and_cache_flash | tokens=512 | 3.697 ms | 0.310 ms | 11.94x |
| index_put(accumulate) | rows=4096 width=256 idx=512 | 6.140 ms | 0.750 ms | 8.18x |
| index_select | rows=4096 width=256 idx=512 | 1.425 ms | 0.246 ms | 5.80x |
| index_put | rows=4096 width=256 idx=512 | 1.285 ms | 0.307 ms | 4.19x |
| index_copy | rows=16384 width=512 idx=4096 | 0.695 ms | 0.272 ms | 2.55x |
| rms_norm | rows=16384 width=4096 | 1.370 ms | 0.825 ms | 1.66x |
| rms_norm | rows=4096 width=1024 | 0.430 ms | 0.304 ms | 1.41x |
| index_copy | rows=4096 width=256 idx=512 | 0.244 ms | 0.270 ms | 0.91x |
| embedding | vocab=128000 dim=512 tokens=4096 | 0.261 ms | 0.306 ms | 0.85x |
| **index_add** | **rows=4096 width=256 idx=512** | **0.591 ms** | **0.891 ms** | **0.66x** |
| embedding | vocab=32000 dim=256 tokens=512 | 0.172 ms | 0.304 ms | 0.57x |
| **index_add** | **rows=16384 width=512 idx=4096** | **0.643 ms** | **1.191 ms** | **0.54x** |

`rms_norm` 本轮改了 kernel 结构（grid 封顶 + 按 `num_programs` 走 tile），0.825 ms
与 0.12 的 0.764 ms 基本一致，循环没有引入可测开销。

**（7）`index_add` 快路径在生产口径下是净亏（待决，见 9）。**

同一个同步闸门，`index_put(accumulate)` 还有 8~53x，`index_add` 却是 0.54~0.66x。
差别不在快路径，在通用基线：上游 `63e7f9d71` 把 `index_add` 的越界校验从
`aminmax` + 两次 `.item()` 换成一个 Triton reduction kernel，通用实现从
11.6 ms 降到 0.64 ms（约 18 倍）。0.12 表里 `index_add` 的 39.20x 是在**旧的慢基线**
上测的；基线一快，闸门同步的固定开销就盖过了 DMA 的收益。

扫了 10 组形状（宽 64~4096、下标 64~16384、行 1024~32768），**没有 crossover**，
全程 0.40x~0.71x。通用开销随搬运量增长，闸门开销只随下标数增长，但通用实现在这
个量级一直是 0.53~1.10 ms，始终比"闸门+kernel"便宜。

试过把两次检查合成一个 kernel + 一次同步（越界下标记到 `num_rows` 处的哨兵槽，
归约同时拿总标记数与单行最大标记数），判定正确（valid/越界/重复 → True/False/False），
省下 0.24~0.74 ms，但仍不够：

| | small | large | huge |
|---|---|---|---|
| 现在两次检查 | 0.707 ms | 0.977 ms | 1.273 ms |
| 合成一次检查 | 0.466 ms | 0.548 ms | 0.537 ms |
| 合成检查 + kernel | 0.610 ms | 0.697 ms | 0.764 ms |
| 通用 | 0.588 ms | 0.643 ms | 0.612 ms |

地板摆在这里：一个空 kernel 加一次 `.item()` 同步本身就要 0.233 ms，加上 0.148 ms
的 DMA kernel 是 0.38 ms，而通用是 0.53~0.65 ms。即便把检查做到零成本也只有
1.4x，而真实检查要 0.47~0.55 ms。**这台机器上、配上游当前的通用实现，
`index_add` 的 TV 快路径赢不回来。**

顺带排除一个假阳性：扫描在 rows=262144 处报 `vector core exception`。把
alloc / has_no_duplicates / 断言 / TV kernel / 通用 / 完整快路径六个阶段各自单进程
重跑，全部通过——是扫描脚本在一个进程里连开 10 组 512 MiB 张量的内存压力，不是
算子缺陷。

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 设备 | Ascend 910B4-1（`ASCEND_RT_VISIBLE_DEVICES=5`） |
| CANN（快路径可用） | 9.1.0-beta.3 (`/data/jianheng/Ascend/cann-9.1.0-beta.3`) |
| CANN（快路径不可用） | 9.1.0 (`/data/jianheng/Ascend/910/cann-9.1.0`) |
| triton / flagtree | 3.5.1，FlagTree `5db164cac` 分支 `common-ir-feat-tensor-view`（2026-10-27 重建） |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post2 |

本文数据于 2026-10-27 在重建后的环境上整套复测。结论未变：快路径仍只在 beta3
可用，9.1.0 下仍是同一条字节码 schema 报错。重测新发现一处真问题（index_add /
index_put 的 `torch.unique` 闸门开销，见第 4.1 节）已修复。

**代码与安装路径不一致。** 重装后 editable `flag_gems` 指向
`/data/jianheng/works/FlagGems`，TensorView 代码在 `/data/jianheng/ppp/FlagGems`。
跑测试需要显式覆盖：

```bash
export PYTHONPATH=/data/jianheng/ppp/FlagGems/src
```

**环境选择会决定走哪条路径，这点很关键。** 用户的 `/data/jianheng/export.sh`
默认指向 CANN 9.1.0，该组合下 TensorView **编译不过**，回退闸门生效、全部测试
走通用实现。本文所有标注「beta3」的结果才是真正验证了快路径的。beta3 环境：

```bash
source /data/jianheng/Ascend/ascend-toolkit/set_env.sh
source /data/jianheng/Ascend/cann/set_env.sh
export ASCEND_TOOLKIT_LATEST_HOME=/data/jianheng/Ascend/cann
export ASCEND_RT_VISIBLE_DEVICES=5
export FLAGTREE_BACKEND=ascend
```

确认当前进程实际走了哪条路径：

```python
from flag_gems.runtime.backend._ascend import tensorview as tv
tv.HAS_TENSORVIEW   # 静态探测结果
tv._state           # 0=未决 1=快路径成功 2=已回退
```

---

## 2. 集成清单

11 个算子，10 个启用快路径，1 个写好但关闭。

| 算子 | 文件 | 视图类型 | 对照实现 |
|---|---|---|---|
| index_select | `ops/index_select.py` | GSV + Partition | Ascend 原实现 |
| index_add / index_add_ | `ops/index_add.py` | GSV + GSV | Ascend 原实现 |
| index_copy / index_copy_ | `ops/index_copy_.py` | GSV | Ascend 原实现 |
| index_put / index_put_ | `ops/index_put.py`（新增） | GSV | 公共实现 |
| embedding（前向） | `ops/embedding.py` | GSV + Partition | Ascend 原实现 |
| rms_norm（前向） | `ops/rms_norm.py` | Partition | Ascend 原实现 |
| fused_add_rms_norm | `fused/fused_add_rms_norm.py` | Partition | Ascend 原实现 |
| reshape_and_cache_flash | `fused/reshape_and_cache_flash.py`（新增） | GSV | 公共实现 |
| concat_and_cache_mla | `fused/concat_and_cache_mla.py`（新增） | GSV ×2 | 公共实现 |
| paged_kv_block_gather | `fused/paged_kv_block_gather.py`（新增） | 3-D GSV | 公共实现（同时新增） |
| cp_gather_indexer_k_quant_cache | `fused/cp_gather_indexer_k_quant_cache.py`（新增） | — **已关闭** | 公共实现 |

`src/flag_gems/fused/paged_kv_block_gather.py` 是随本次一起新增的公共参考实现，
既做 fallback 也做性能对照基线。

注册改动：`_ascend/ops/__init__.py`、`_ascend/deprecated/__init__.py`、
`src/flag_gems/fused/__init__.py`、`conf/operators.yaml`。

上表的 `fused/` 路径是第二轮的布局。第四轮起这 11 个文件都在
`_ascend/deprecated/` 下，原因见 0.14 节。

---

## 3. 回退设计

TensorView 需要特定的 flagtree 构建 **和** 匹配的 CANN 版本，二者都不能只靠
Python 命名空间判断出来——一个 `bishengir-opt` 与本地 `triton-mlir-opt`
字节码不兼容的 toolkit 能通过所有 `hasattr` 检查，然后在编译期才报错。所以分两层：

**静态探测** `HAS_TENSORVIEW`：探测三个构造器 + `tile_load`/`tile_store` +
Ascend `UB` 地址空间是否可导入。为 False 时 `@triton.jit` 快路径 kernel 根本不
定义，模块退化成通用实现的别名。

注意不能用 vendor 级的 `HAS_TLE`：Ascend 把那个开关关着，即使 TLE 可导入也是
False，所以这里直接探测具体原语（和 `ops/glu.py` 探测 DSA 助手的做法一致）。

**首次调用 try/except** `tensorview_call`：跑一次快路径，只吸收编译错误
（按类名/模块匹配 `MLIRCompilationError` / `CompilationError`，避免在没有这些
异常类的后端 import 失败），命中后进程级永久降级，日志打一条 warning。
其他异常一律上抛——TensorView kernel 里的数值或形状 bug 仍然会以测试失败的形式
暴露出来，不会被通用实现悄悄盖住。

所以一个不支持的 toolkit 的代价是**每进程一次失败编译**，而不是每次调用一次。

CANN 9.1.0 下实测的回退行为（这就是设计要应对的场景）：

```
error: expected mlir::hivm::TCoreTypeAttr, but got: loc(".../index_select.py":161:8)
note: in bytecode version 6 produced by: MLIR22.0.0git
WARNING GEMS_ASCEND INDEX_SELECT: the TensorView path does not compile against
this flagtree/CANN combination (...); using the generic implementation for the
rest of this process.
→ HAS_TENSORVIEW True / _state 2 / 结果正确
```

**每个算子还有一层 eligibility 闸门**，不符合的形状直接走通用实现而不进
`tensorview_call`。值得单独说的几条：

- `index_add` / `index_put(accumulate=True)` 要求 `index` 无重复
  （`torch.unique(index).numel() == index.numel()`）：块 DMA 的读-改-写在重复
  下标上会丢更新。`index_copy_` 则不需要这条——纯写入、last-writer-wins 与
  torch 语义一致。
- `index_add` 排除 bfloat16（通用路径用 fp32 中转）。
- 两个 norm 要求 `_tv_block_m(width) > 1`，理由见第 6 节。
- `concat_and_cache_mla` 要求 `rank % rope == 0` 且 `kv_cache.size(2) == rank + rope`
  （rope 半边存在稠密 tile 下标 `RANK // ROPE` 上）。
- `reshape_and_cache_flash` 要求 `kv_cache_dtype == "auto"`（fp8 走通用）加两个
  cache 的精确紧致 stride 检查。

---

## 4. UB 容量：性能测试才暴露的真 bug

tile 要暂存在 192 KB 的 Unified Buffer 里。固定 `BLOCK_M=32` 时，宽行会超限，
并且**超限是编译期失败而不是降级**，报错在 `ConvertTritonIRToLinalgIR`。

受影响形状（修复前全部编译失败）：

| 算子 | 失败阈值 |
|---|---|
| rms_norm | width ≥ 1024 |
| fused_add_rms_norm | width ≥ 1024 |
| reshape_and_cache_flash | row ≥ 1024 |
| index_select / embedding | width ≥ 4096 |
| index_add | width ≥ 1024 |
| index_copy / index_put | 同量级 |
| paged_kv_block_gather | payload ≥ 128 KB |
| concat_and_cache_mla | rank ≥ 2048 |

修复：`tensorview.py` 里加共享助手

```python
def tile_rows_for_ub(row_bytes, live_tiles, max_rows):
    """Largest power-of-two row count per tile whose staged tiles fit in UB."""
```

按行宽和同时存活的 tile 数反推每 tile 行数，预算取 UB 的一半（编译器还要留
double buffering 和 kernel 自己物化的中间量的空间）。各算子的 `live_tiles`
按 fp32 提升后的实际占用算：

| 算子 | live_tiles | 说明 |
|---|---|---|
| index_select / index_copy / embedding | 1 | 单个 gather tile |
| index_add | 2 | 两个 fp32 tile |
| index_put | 1 或 2 | `accumulate` 时为 2 |
| rms_norm | 2 | tile + 归一化结果，均 fp32 |
| fused_add_rms_norm | 3 | x、residual、和 |
| reshape_and_cache_flash | 2 | — |

`paged_kv_block_gather` 的 tile 形状由 block table 决定、没有 BLOCK 可缩，
改成超限直接判不合格走通用实现。

**这点现有测试完全抓不到**——FlagGems 相关用例宽度都 ≤512。绿色测试套件不能
证明快路径在生产宽度下能编译。修复后的宽形状回归（beta3，`tv_state=1`）：

```
index_select W=512/1024/2048/4096/8192   match=True
embedding    D=512/1024/4096             match=True
reshape_and_cache_flash row=256/512/1024/2048   编译通过
index_add (4096,256)/(2048,1024)/(1024,4096)    match=True
index_put (16384,128)/(4096,256)/(8192,512) × acc∈{False,True}  match=True
```

### 4.1 `torch.unique` 闸门：重测才暴露的第二个真 bug

重测时把闸门也算进计时（之前只测了 kernel），index_add 的加速比从报告的
116x 掉到 6.58x，且标准差异常大（±0.021）。拆开量：

| 形状 | `torch.unique` 闸门 | kernel | 合计 |
|---|---|---|---|
| (4096,256) idx=1024 | 0.2925 | 0.0121 | 0.3151 |
| (2048,1024) idx=512 | 0.1700 | 0.0101 | 0.1931 |

**闸门比它守护的 kernel 贵 25 倍**，因为 `torch.unique` 要排序。替代方案实测：

| 做法 | (4096,1024) | (16384,2048) |
|---|---|---|
| `torch.unique` | 0.2978 | 0.5275 |
| `torch.bincount` | 0.1655 | 0.4579 |
| `scatter_` 打标记 | **0.0765** | **0.1912** |

改成 `tensorview.py` 里的共享助手 `has_no_duplicates(index, num_rows)`：
scatter 一遍 1，每个被点到的行只会被标一次，标记之和等于 index 长度当且仅当无重复。
不用 `index_add_` 做这件事——拿有竞争风险的 add 去检测 add 的竞争，逻辑上是循环。

修复后（beta3）：

| 算子 | 形状 | 修前 | 修后 |
|---|---|---|---|
| index_add | (4096,256) idx=1024 | 6.58x | **20.43x** |
| index_add | (2048,1024) idx=512 | 19.99x | **73.37x** |

`index_put(accumulate=True)` 用的是同一个闸门，一并换掉。回归：
`test_index_add.py` 56 passed，`test_index_put.py -k dtype0` 37 passed。

重复下标语义也单独验过：`dup=True` 时两个算子都正确退到通用实现。其中
`index_put(acc=False)` + 重复下标与 CPU 参考不一致，但**通用实现同样不一致**
——last-writer-wins 在 torch 里本就非确定性，每个写入行都等于它两个候选源行之一，
不是闸门改动引入的。

---

## 5. 功能测试结果（beta3，快路径真实生效）

| 测试套件 | 结果 |
|---|---|
| `tests/test_index_select.py` | 18 passed |
| `tests/test_index_add.py` | 56 passed |
| `tests/test_index_copy_.py` | 19 failed, 5 passed（逐例隔离后 10 passed / 2 failed，是 beta3 的 aclnn bug，见 6.2） |
| `tests/test_index_put.py` -k dtype0 | 46 passed（排除 1 个 poison 用例，见 6.1） |
| `tests/test_embedding.py` | 504 passed, 27 skipped |
| `tests/test_rms_norm.py` | 9 passed |
| `tests/test_fused_add_rms_norm.py` | 9 passed |
| `tests/test_add_rms_norm.py` | 9 passed |
| `tests/test_reshape_and_cache_flash.py` | 72 passed |
| `tests/test_concat_and_cache_mla.py` | 6 failed（用例写死 CUDA，见 6.3） |
| `tests/test_cp_gather_indexer_k_quant_cache.py` | 10 skipped（用例 gate 在 CUDA，见 6.4） |

2026-10-27 复测全部一致：`test_index_copy_.py` 逐例隔离仍是 dim=1/shape2 的 4 例失败
（其余 20 例通过），`test_index_put.py` 的 poison 用例仍报同一个 `coreDim 131072`。

后两个算子没有可用的上游测试，直接与各自的通用实现对比验证（beta3，`tv_state=1`）：

```
concat_and_cache_mla  tokens=1024 rank=512 rope=64   match=True
concat_and_cache_mla  tokens=4096 rank=256 rope=64   match=True
paged_kv_block_gather nb=512  seqs=64  bps=8         match=True
paged_kv_block_gather nb=1024 seqs=128 bps=8         match=True
```

---

## 6. 失败用例逐条说明

以下全部为**既有问题，与 TensorView 改动无关**，每条都已验证。两个真实失败的
归属已经做了隔离复现（见 6.1 / 6.2 的「归属」小节）：一个是**通用实现的
launch 配置 bug**，另一个是**CANN beta3 的 aclnn 算子 bug**，都不在
TensorView 代码里。

### 6.1 `test_index_put.py` — 1 个用例拖垮 45 个

直接跑整个套件是 45 failed，但这不是 45 个独立失败。根因是单个用例的
launch 参数越界，device 报错后污染整个进程，其余用例级联失败。

失败用例：

```
tests/test_index_put.py::test_index_put_acc_false[dtype0-input_shape1-indices_shape1-values_shape1-False]
tests/test_index_put.py::test_index_put__acc_false[dtype0-input_shape1-indices_shape1-values_shape1-False]
```

形状 `input_shape=(524288,)`, `indices_shape=((262144,),)`, `values_shape=(262144,)`。报错：

```
Invalid_Argument(EE1003): Starting the compute task of the corresponding operator
failed because value 131072 for parameter coreDim is invalid.
Expected value: less than or equal to 65535.
        The argument is invalid. Reason: rtKernelLaunch execution failed.
RuntimeError: ... the current working operator name is _index_put_jit_function.
```

**归属：通用实现（`src/flag_gems/ops/index_put.py`）的 launch 配置 bug。**

算得很清楚：`src/flag_gems/ops/index_put.py:179` 的 grid 是
`triton.cdiv(M, meta['BLOCK_SIZE0'])`，`M = indices[0].numel() = 262144`，
默认 `BLOCK_SIZE0 = 2`（第 88 行），于是 grid.x = 262144 / 2 = **131072**，
正好是报错里的那个值，超过 Ascend 的 65535 上限一倍。

绕开测试、直接调通用实现即可复现，与 TensorView 无关：

```python
from flag_gems.ops.index_put import index_put_ as gen   # 公共实现，非 Ascend 覆盖
gen(inp, [idx], val, False)
# → Invalid_Argument(EE1003): KernelLaunch failed because value 131072
#   for parameter coreDim is invalid. Expected value: less than or equal to 65535.
```

两个 CANN（9.1.0 与 beta3）下都一样失败，说明是纯粹的 host 侧 grid 计算问题，
不是编译器或 toolkit 的问题。上游测试用例自己的注释就写了这个限制
（「grid.x = M / BLOCK_SIZE0 must stay within the launch limit, which is 65535
on some backends」），即该后端需要一个会把 grid.x 分块的 kernel，而通用实现
没有做这件事。

该形状也不会进快路径（1-D 输入，我的闸门要求 `ndim >= 2`）。修法应是在通用
实现里对 grid.x 分块或抬高 `BLOCK_SIZE0`，属于本次提交范围之外。

级联证据：

```
全套件              → 45 failed
排除这 1 个用例      → 11 passed（acc_false 其余全过）
排除 2 个 _acc_false → 46 passed, 0 failed
单独跑其中任一其他用例 → 1 passed
```

同时与上游 master 对比，失败集合**逐字节相同**（`diff` 为空，各 45 条）。

### 6.2 `test_index_copy_.py` — dim=1 的 2 个用例

整套件 19 failed 同样是级联（device fault 污染进程）。逐例隔离后（beta3）：

```
test_index_copy  dim=1 shape2 → FAILED
test_index_copy_ dim=1 shape2 → FAILED
其余 10 个（dim=0 全部 shape，dim=1 shape0/shape1） → 全部 passed
```

`shape2 = (200, 40999, 3)`。报错：

```
RuntimeError: ... the current working operator name is aclnnStack.
[Error]: The vector core execution is abnormal.
AclNN_Runtime_Error(EZ9903): aclrtLaunchKernelWithHostArgs failed, return: 507035
        rtsLaunchKernelWithHostArgs execution failed, reason=vector core exception
```

**归属：CANN 9.1.0-beta.3 的 `aclnn` 算子 bug，既不在 FlagGems 也不在 Triton。**

开 `ASCEND_LAUNCH_BLOCKING=1` 后故障归到测试第 41 行，也就是**参考**实现那一行
（`TO_CPU` 关闭时 `to_reference` 不搬到 CPU，所以 `torch.index_copy` 是在 NPU 上
跑 torch_npu 自己的 aclnn 算子）。把 FlagGems 完全摘掉仍然能复现：

```python
import torch, torch_npu          # 没有 import flag_gems，没有 triton
inp = torch.randn((10, 1024, 3), dtype=torch.float16, device='npu')
src = torch.randn((10, 1024, 3), dtype=torch.float16, device='npu')
idx = torch.randperm(1024).npu()
torch.index_copy(inp, 1, idx, src); torch.npu.synchronize()
# → MTE accesses an invalid GM address ... aivec error, error code = 0x800000
```

逐步同步确认分配与 `randperm` 都正常，故障点就是 `index_copy` 本身：

```
step1 randperm OK / step2 inp OK / step3 src OK / step4 → FAULT
```

触发条件（`(10, M, 3)`，dim=1，fp16）：

| 条件 | 结果 |
|---|---|
| dim=0 / dim=2（同形状） | OK |
| dim=1，M ≤ 320 | OK |
| dim=1，M ≥ 328 | FAULT |
| dim=1，M=512 但只取 K=16/64 个下标 | OK |
| dim=1，M=512 取 K=320/352 个下标 | FAULT |
| 2-D `(200, 40999)` dim=1 | OK |

即「3-D + 中间维 + 下标数超过约 320」才触发，报错是
`MTE accesses an invalid GM address`（越界访存）。

**决定性证据是换 CANN 就好了。** 同一段代码、同一形状：

| CANN | M=320 | M=352 | M=1024 | `(200,40999,3)` |
|---|---|---|---|---|
| 9.1.0-beta.3 | OK | FAULT | FAULT | FAULT |
| 9.1.0 | OK | OK | OK | OK |

所以这是 beta3 里 `aclnnIndexCopy` 这条路径的回归，与算子实现无关。

另外两点佐证这不是我的改动：日志显示走的是
`index_copy_.py:214 GEMS_ASCEND INDEX_COPY`，即**通用**路径（`dim=1` 被我的闸门
拒绝，只接受 `dim % ndim == 0`）；该文件 diff 为 `146 insertions, 2 deletions`，
两处删除只是把 `index_copy`/`index_copy_` 改名为
`_index_copy_generic`/`_index_copy__generic`，通用实现函数体与上游逐字节一致。

### 6.3 `test_concat_and_cache_mla.py` — 6 个用例全部失败

```
FAILED test_concat_and_cache_mla[auto-cuda:0-0-dtype0-8-16-42-64-512]
FAILED test_concat_and_cache_mla[auto-cuda:0-0-dtype1-8-16-42-64-512]
FAILED test_concat_and_cache_mla[auto-cuda:0-0-dtype2-8-16-42-64-512]
FAILED test_concat_and_cache_mla[auto-cuda:1-0-dtype0-8-16-42-64-512]
FAILED test_concat_and_cache_mla[auto-cuda:1-0-dtype1-8-16-42-64-512]
FAILED test_concat_and_cache_mla[auto-cuda:1-0-dtype2-8-16-42-64-512]
```

报错：

```
AssertionError: Torch not compiled with CUDA enabled
/usr/local/python3.11.15/.../torch/cuda/__init__.py:403
```

用例第 27 行 `CUDA_DEVICES = [f"cuda:{i}" for i in ...]` 写死 CUDA 设备，在 NPU
上必然失败，与实现无关。该算子的正确性用第 5 节的直接对比验证。

### 6.4 `test_cp_gather_indexer_k_quant_cache.py` — 10 个全部跳过

```
10 skipped
```

用例第 29 行 `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)`，
在 NPU 上整体跳过，从未在本机运行过。

### 6.5 cp_gather_indexer_k_quant_cache 快路径已关闭

`_TV_ENABLED = False`，模块直接转发通用实现。三次尝试均失败后停止：

1. `strides=[1, 1]` 扁平偏移视图不下降
   （`ConvertTritonIRToLinalgIR` / `PassManager::run failed`）。
2. 改用 tile 下标寻址，device 报 `vector core exception`。
3. 用 `numel() // head_dim` 算 extent 少算了 padding 行；改成按完整分配算
   仍然 fault。

已排除的可能：padded-row 视图本身能下降且单独读取正确；行下标在声明的 extent
内；scale 切片几何一致；去掉 scale 存储只留 gather 半边仍然 fault，所以不是
scale 运算的问题。

更关键的是，即使修好收益也有限：上游测试的 10 个形状里有 9 个本来就不合格
——内联的 fp32 scale 让 block-row stride 不是 `head_dim` 的整数倍，行无法按
tile 下标寻址。另外该算子的**通用实现在本机也编译不过**（已在上游 master 上
确认），也就是说它在这台机器上从来没能跑起来过。

---

## 7. 性能数据

方法与 FlagTree 基准一致：`do_bench_npu`，warmup 25 / active 50，5 次 profiler
重复，取 5 个均值的中位数，±为样本标准差，单位 ms。对照方是**被覆盖的那个实现**
（有 Ascend 原实现就用它，没有就用公共实现），两边都先对同一参考验过再计时。

### 行 gather/scatter 类

| 算子 | 形状 | TensorView | 对照 | 加速 |
|---|---|---|---|---|
| index_select | (8192,4096) idx=1024 | 0.012607±0.000098 | 6.093693±0.000279 | **483.36x** |
| reshape_and_cache_flash | tokens=1024 row=1024 | 0.014629±0.000072 | 6.844550±0.001648 | **467.87x** |
| reshape_and_cache_flash | tokens=4096 row=512 | 0.040636±0.000161 | 13.504710±0.002372 | 332.33x |
| reshape_and_cache_flash | tokens=1024 row=256 | 0.012088±0.000025 | 1.725600±0.000244 | 142.75x |
| index_select | (32000,256) idx=4096 | 0.018482±0.000042 | 2.477560±0.000238 | 134.06x |
| index_put(acc) | (4096,256) idx=1024 | 0.094238±0.000062 | 11.190043±0.010194 | 118.74x |
| index_put(acc) | (16384,128) idx=2048 | 0.219327±0.000034 | 21.798032±0.015169 | 99.39x |
| index_select | (2048,768) idx=512 | 0.006900±0.000035 | 0.620166±0.000106 | 89.88x |
| index_add | (2048,1024) idx=512 | 0.051908±0.000094 | 3.808643±0.000400 | 73.37x |
| index_select | (4096,256) idx=1024 | 0.007386±0.000021 | 0.401995±0.000107 | 54.43x |
| index_put | (16384,128) idx=2048 | 0.020645±0.000075 | 3.022958±0.000617 | 146.43x |
| index_put | (4096,256) idx=1024 | 0.013755±0.000050 | 2.011955±0.000360 | 146.27x |
| index_add | (4096,256) idx=1024 | 0.094693±0.000149 | 1.934242±0.000575 | 20.43x |
| index_copy | (16384,128) idx=2048 | 0.020735±0.000076 | 0.158702±0.001328 | 7.65x |
| index_copy | (4096,256) idx=1024 | 0.013771±0.000031 | 0.081774±0.000929 | 5.94x |

两个 index_add 行的时间含 `has_no_duplicates` 闸门（约 0.03–0.08ms）；index_put
的 `acc=False` 不需要该闸门，所以比 `acc=True` 快一个量级。

### norm 与其他

| 算子 | 形状 | TensorView | 对照 | 加速 |
|---|---|---|---|---|
| rms_norm | rows=4096 width=512 | 0.028756±0.000025 | 0.305172±0.004925 | 10.61x |
| rms_norm | rows=8192 width=1024 | 0.078799±0.000947 | 0.596112±0.009343 | 7.56x |
| rms_norm | rows=2048 width=4096 | 0.078755±0.001153 | 0.153487±0.002101 | 1.95x |
| rms_norm | rows=1024 width=8192 | 0.078792±0.001188 | 0.079302±0.001462 | 1.01x（已闸门排除） |
| concat_and_cache_mla | tokens=4096 rank=256 rope=64 | 0.040124±0.000045 | 0.301421±0.004182 | 7.51x |
| paged_kv_block_gather | blocks=1024 seqs=128 bps=8 | 0.011938±0.000139 | 0.076171±0.001209 | 6.38x |
| fused_add_rms_norm | rows=4096 width=512 | 0.047102±0.000023 | 0.297838±0.002438 | 6.32x |
| concat_and_cache_mla | tokens=1024 rank=512 rope=64 | 0.013009±0.000026 | 0.077137±0.001274 | 5.93x |
| embedding | vocab=128256 tokens=2048 dim=512 | 0.013056±0.000049 | 0.077001±0.001406 | 5.90x |
| fused_add_rms_norm | rows=8192 width=1024 | 0.114350±0.000113 | 0.609558±0.006930 | 5.33x |
| embedding | vocab=32000 tokens=1024 dim=256 | 0.007396±0.000032 | 0.038657±0.000625 | 5.23x |
| paged_kv_block_gather | blocks=512 seqs=64 bps=8 | 0.008197±0.000148 | 0.038518±0.000513 | 4.70x |
| fused_add_rms_norm | rows=2048 width=4096 | 0.079274±0.000958 | 0.151926±0.002521 | 1.92x |

共 28 行，比值 1.0x 到 483x，中位数约 10x。收益最大的是行 gather/scatter 类
（index_select / index_put / index_add / reshape_and_cache_flash），正是把逐元素
访存换成整行块 DMA 的场景。norm 类 5–10x，收益来自把多行批进一个 tile。

### rms_norm 的 crossover

| width | 加速 | BLOCK_M |
|---|---|---|
| 512 | 10.61x | 32 |
| 1024 | 7.56x | 16 |
| 4096 | 1.95x | 4 |
| 8192 | 1.01x | 1 |

收益严格跟着 `BLOCK_M` 走。width=8192 时 `BLOCK_M` 已被 UB 预算压到 1，快路径
和逐行的通用 kernel 做的是同一件事，没有任何收益。因此两个 norm 的闸门里都加了
`_tv_block_m(width) > 1`，这类形状直接走通用实现，不白绕一圈。

基准脚本 `bench_tensorview_ascend.py` 是临时验证工具，测完已删除。
`cp_gather_indexer` 未计入以上数据（已关闭，且其通用实现在本机也编译不过）。

---

## 8. 失败归属汇总

| 失败 | 归属 | 证据 |
|---|---|---|
| index_put `input_shape1`（2 例） | **通用实现 host 侧 grid 计算** | grid.x = 262144/2 = 131072 > 65535；绕开测试直调通用实现即复现；两个 CANN 都失败 |
| index_copy_ `dim=1-shape2`（2 例） | **CANN beta3 的 aclnn 算子** | 纯 `torch.index_copy`（无 flag_gems / 无 triton）即复现；换 CANN 9.1.0 即正常 |
| concat_and_cache_mla（6 例） | **测试用例** | 第 27 行写死 `cuda:0`/`cuda:1` |
| cp_gather_indexer（10 skipped） | **测试用例** | 第 29 行 gate 在 `torch.cuda.is_available()` |
| TensorView 整体不可用（CANN 9.1.0） | **编译器 / toolkit 版本错配** | `expected mlir::hivm::TCoreTypeAttr`，本地 `triton-mlir-opt` 与该 CANN 的 `bishengir-opt` 字节码 schema 不一致 |
| `torch.randperm` 编译失败（新增） | **新 triton 的标量范围检查 + 上游 randperm** | 上游 master 同样失败；`randperm.py:116` 的 `ik \| 0x8000` 在 int16 分支超范围，日志 `log/20261027_randperm_int16_scalar_range.log` |

**没有一条归到 TensorView 算子实现上。** 归到我这边的有两条，都已修复并回归验证：
第 4 节的 UB 容量、第 4.1 节的 `torch.unique` 闸门开销。

### 8.1 环境更新带来的新回归：`torch.randperm`

重建后的 triton 对 `tl.where` 的标量常量按 `k.dtype` 做范围检查，
`src/flag_gems/ops/randperm.py:116` 的 int16 分支 `ik | 0x8000` 直接报
`ValueError: Scalar 32768 is out of range for type int16`（`ik` 虽已 `.to(tl.int64)`，
常量仍按原 dtype 校验）。

两个 CANN 都复现，`ppp/FlagGems` 与上游 `works/FlagGems` 表现一致，
且该文件不在 TensorView 提交的 diff 内，所以与本次改动无关。影响面是任何
`flag_gems.enable()` 后调 `torch.randperm` 的脚本——测试里构造随机下标很常用，
本次验证脚本都改成在 CPU 上生成下标再 `.npu()` 绕开。

## 9. 待决问题

**两个 CANN 各坏一样东西，目前没有哪个能跑完全绿的套件：**

| | TensorView 快路径 | `aclnnIndexCopy`（测试参考路径） |
|---|---|---|
| CANN 9.1.0（`export.sh` 默认） | ✗ 编译失败，静默回退 | ✓ 正常 |
| CANN 9.1.0-beta.3 | ✓ 正常 | ✗ 越界访存 fault |

所以 index_copy_ 的那 2 个用例只能在「快路径可用」的环境下失败，而在「快路径
不可用」的环境下通过——两者无法同时满足，除非把编译器扩展针对 CANN 9.1.0
重建。这也是推荐的解法：重建后既能用快路径，又能避开 beta3 的 aclnn 回归。

在此之前，默认环境下跑 FlagGems 测试会**静默走通用实现**——结果正确、测试全绿，
但完全没有验证 TensorView。跑之前先确认 `tv._state`。`export.sh` 保持原样未改动。

**`index_add` 的快路径是否还要保留（需要决定）。**

0.16（7）的数据是明确的：生产口径下 0.54x~0.66x，10 组形状无 crossover，合并检查
与零成本检查两个方向都算过，都赢不回来。原因是上游 `63e7f9d71` 把通用实现提速了
约 18 倍，而快路径必须付的那次闸门同步（读-改-写要求下标无重复）是固定开销。

三条路，倾向第一条：

1. **摘掉 `index_add` / `index_add_` 的覆盖**，`ops/__init__.py` 不再导出，实现文件
   保留或删除。用户拿到的是更快的通用实现，TV 层只留真正有收益的算子。代价是 17 个
   移植算子少一个。
2. 保留实现但让闸门恒假。契约上和（1）等价，多留一份不会执行的代码。
3. 原样保留。会给用户带来约 1.8 倍的回归，不建议。

这是缩减移植范围的决定，按约定留给你定；在此之前文件保持现状（已修的越界断言
是独立的正确性修复，与这个决定无关，无论怎么选都该留着）。

另外两条待处理（都不在 TensorView 范围内）：

- `torch.randperm` 在新 triton 下编译不过（第 8.1 节），上游问题，需要改
  `ops/randperm.py` 的 int16 分支。
- editable 安装指向 `works/FlagGems`，跑本分支必须带
  `PYTHONPATH=/data/jianheng/ppp/FlagGems/src`，否则测的是上游代码。
