---
title: "为什么 ptr.load 没有直接的 LLVM Dialect 转换"
weight: 95
---

# 为什么 `ptr.load` 没有直接的 `LLVM::LoadOp` 转换

## 背景

在使用 MLIR `ptr` dialect 时，用户可能会注意到 `ptr.load`（即
`mlir::ptr::LoadOp`）**没有**明显的到 LLVM dialect `LLVM::LoadOp` 的
lowering，也没有一个独立的 `-convert-ptr-to-llvm` pass。
本文档记录了上游社区（llvm/llvm-project）中解释这一设计的相关证据。

## 搜索摘要

在 `llvm/llvm-project` GitHub 仓库（issues 和 pull requests）中检索了以下关键词：

- `ptr.load`
- `PtrToLLVM`
- `convert-ptr-to-llvm`
- `populatePtrToLLVMConversionPatterns`
- `registerConvertPtrToLLVMInterface`
- `Ptr dialect`
- `LLVM::LoadOp`

## 关键发现

### 1. `ptr.load` 和 `ptr.store` 在引入时就刻意没有提供 LLVM Dialect 转换

**PR [#156093](https://github.com/llvm/llvm-project/pull/156093)**
*"[mlir][ptr] Add load and store ops."*（2025-09-01 合入，作者：`fabianmcg`）

> "This patch adds the load and store operations to the ptr dialect.
> It's **future work** to implement SROA and Mem2Reg interfaces,
> **as well as conversion to LLVM**, and add alias information."
>
> （本补丁在 ptr dialect 中添加了 load 和 store 操作。
> 实现 SROA 和 Mem2Reg 接口，**以及到 LLVM 的 conversion**，以及添加别名信息，
> 均为**未来工作**。）

这是最直接的上游证据：引入 `ptr.load` / `ptr.store` 的维护者明确将 LLVM
Dialect 的 conversion 标注为"未来工作"，而不是遗漏。

### 2. 已有的 PtrToLLVM 转换基础设施是"权宜之计"，且早于 `ptr.load` 引入

**PR [#156053](https://github.com/llvm/llvm-project/pull/156053)**
*"[mlir][ptr] Add conversion to LLVM for all existing `ptr` ops"*
（2025-08-29 合入，作者：`fabianmcg`）

> "This patch adds conversion to LLVM for all existing pointer ops.
> This is a **stop gap measure** to allow users to use the `ptr` dialect now.
> In the future some of these conversions will be removed, and added as
> translations, for example `ptradd`."
>
> （本补丁为所有现有指针 op 添加了到 LLVM 的 conversion。
> 这是一个**权宜之计**，以便用户现在就能使用 `ptr` dialect。
> 未来部分 conversion 将被移除，转为 translation，例如 `ptradd`。）

该 PR 在 `ptr.load` / `ptr.store` 引入**前两天**合入。它只覆盖了
`FromPtrOp`、`GetMetadataOp`、`PtrAddOp`、`ToPtrOp`、`TypeOffsetOp`，
并不包含 load/store（彼时还不存在）。
"权宜之计"的措辞也表明团队的长期方向是**直接 LLVM IR translation**，
而非通过 LLVM Dialect conversion。

### 3. LLVM IR translation（非 Dialect conversion）为 `ptr.load` 提供了支持

**PR [#156355](https://github.com/llvm/llvm-project/pull/156355)**
*"[mlir][ptr] Add translations to LLVMIR for ptr ops."*
（2025-09-03 合入，作者：`fabianmcg`）

该 PR 实现了 `PtrToLLVMIRTranslation`，将 `ptr.load` 直接翻译为
`llvm::LoadInst`（完整支持 `volatile`、`atomic`、`nontemporal`、
`invariant` 和 `syncscope`），**绕过 LLVM Dialect，直接生成 LLVM IR**。

配套的 **PR [#156333](https://github.com/llvm/llvm-project/pull/156333)**
*"[mlir][LLVM|ptr] Add the `#llvm.address_space` attribute, and allow `ptr` translation"*
（2025-09-02 合入）添加了 `#llvm.address_space` 属性，为 `!ptr.ptr` 类型提供
了翻译所需的地址空间语义。

### 4. 设计方向：优先使用 translation 而非 dialect conversion

**PR [#157347](https://github.com/llvm/llvm-project/pull/157347)**
（2025-09-14 合入）将 `PtrToLLVM` 文件中所有 `convert*` 命名改为
`translate*`，明确表明官方意图：从 dialect conversion 迁移到直接 translation。

## 结论

`ptr.load → LLVM::LoadOp` conversion 的缺失是**有意设计，而非疏忽**：

1. `ptr.load` 引入时（PR #156093），其 LLVM lowering 被明确标为"未来工作"。
2. 官方首选路径是**直接 LLVM IR translation**（通过 `PtrToLLVMIRTranslation`），
   而非先 lower 到 LLVM Dialect op。
3. 现有的 `PtrToLLVM` Dialect conversion 基础设施被定性为"权宜之计"，
   仅覆盖较早期的 ptr ops，不包含 load/store。

## 当前状态（研究时间节点）

| 路径 | `ptr.load` 支持情况 |
|------|---------------------|
| LLVM IR translation（`mlir-translate`） | ✅ 已支持（PR #156355，2025-09-03 合入） |
| LLVM Dialect conversion（`mlir-opt --convert-to-llvm`） | ❌ 暂不支持（未来工作，见 PR #156093） |

## 临时方案

如果你现在就需要 `ptr.load → llvm.load`，可以自行编写一个小的
`ConvertOpToLLVMPattern<ptr::LoadOp>`，将该 op 的属性（alignment、volatile、
atomic ordering、syncscope、nontemporal、invariant）映射到对应的
`LLVM::LoadOp` 属性。或者，如果 pipeline 最终产出 LLVM IR，可以使用
`mlir-translate` 路径。

## 进一步提问的渠道

- **LLVM Discourse（MLIR 分类）**：https://discourse.llvm.org/c/mlir/
  — 适合设计类问题和"何时落地"的讨论。
- **GitHub issue tracker**（`llvm/llvm-project`）：如有需求，可提 feature request
  以获得上游关注。

## 参考资料

| PR | 标题 | 时间 |
|----|------|------|
| [#73057](https://github.com/llvm/llvm-project/pull/73057) | [mlir] Ptr dialect（原型） | 2023-11-22 |
| [#86860](https://github.com/llvm/llvm-project/pull/86860) | [mlir][Ptr] 初始化 Ptr dialect 与 `!ptr.ptr` 类型 | 2024-06-27 合入 |
| [#86870](https://github.com/llvm/llvm-project/pull/86870) | [mlir][Ptr] 添加 `MemorySpaceAttrInterface` | 2025-03-19 合入 |
| [#156053](https://github.com/llvm/llvm-project/pull/156053) | [mlir][ptr] 为所有现有 ptr ops 添加 LLVM conversion（权宜之计） | 2025-08-29 合入 |
| [#156093](https://github.com/llvm/llvm-project/pull/156093) | [mlir][ptr] 添加 load 和 store ops（明确 conversion 为未来工作） | 2025-09-01 合入 |
| [#156333](https://github.com/llvm/llvm-project/pull/156333) | [mlir][LLVM\|ptr] 添加 `#llvm.address_space` 属性，允许 ptr translation | 2025-09-02 合入 |
| [#156355](https://github.com/llvm/llvm-project/pull/156355) | [mlir][ptr] 为 ptr ops 添加 LLVM IR translation（含 ptr.load） | 2025-09-03 合入 |
| [#157347](https://github.com/llvm/llvm-project/pull/157347) | [mlir][ptr] 添加 ConstantOp（rename convert→translate） | 2025-09-14 合入 |
| [RFC on Discourse](https://discourse.llvm.org/t/rfc-ptr-dialect-modularizing-ptr-ops-in-the-llvm-dialect/75142) | [RFC] `ptr` dialect 与 LLVM dialect 指针 op 模块化 | 2023 |
