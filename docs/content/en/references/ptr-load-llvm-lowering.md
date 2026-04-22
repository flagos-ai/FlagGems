---
title: "Why ptr.load Has No Direct LLVM Dialect Conversion"
weight: 95
---

# Why `ptr.load` Has No Direct `LLVM::LoadOp` Conversion

## Background

When working with the MLIR `ptr` dialect, users may notice that `ptr.load`
(the `mlir::ptr::LoadOp`) does **not** have an obvious lowering to the LLVM
dialect's `LLVM::LoadOp`, nor is there a standalone `-convert-ptr-to-llvm`
pass.  This document records the upstream evidence explaining why.

## Search Summary

The following keywords were searched in the `llvm/llvm-project` GitHub
repository (issues and pull requests):

- `ptr.load`
- `PtrToLLVM`
- `convert-ptr-to-llvm`
- `populatePtrToLLVMConversionPatterns`
- `registerConvertPtrToLLVMInterface`
- `Ptr dialect`
- `LLVM::LoadOp`

## Key Findings

### 1. `ptr.load` and `ptr.store` were introduced without LLVM dialect conversion — deliberately

**PR [#156093](https://github.com/llvm/llvm-project/pull/156093)**
*"[mlir][ptr] Add load and store ops."*
(merged 2025-09-01, author: `fabianmcg`)

> "This patch adds the load and store operations to the ptr dialect.
> It's **future work** to implement SROA and Mem2Reg interfaces,
> **as well as conversion to LLVM**, and add alias information."

This is the clearest upstream evidence: the maintainer who introduced
`ptr.load` and `ptr.store` explicitly flagged their LLVM dialect conversion
as future work, not an oversight.

### 2. The earlier general PtrToLLVM conversion was a "stop gap" — and predates `ptr.load`

**PR [#156053](https://github.com/llvm/llvm-project/pull/156053)**
*"[mlir][ptr] Add conversion to LLVM for all existing `ptr` ops"*
(merged 2025-08-29, author: `fabianmcg`)

> "This patch adds conversion to LLVM for all existing pointer ops.
> This is a **stop gap measure** to allow users to use the `ptr` dialect now.
> In the future some of these conversions will be removed, and added as
> translations, for example `ptradd`."

This PR was merged **two days before** `ptr.load` and `ptr.store` were added.
It covers `FromPtrOp`, `GetMetadataOp`, `PtrAddOp`, `ToPtrOp`, and
`TypeOffsetOp`, but not load/store (which didn't exist yet).
The "stop gap" language also signals the team's intent: the preferred
long-term path is **direct LLVM IR translation**, not dialect conversion.

### 3. LLVM IR translation (not dialect conversion) was added for `ptr.load`

**PR [#156355](https://github.com/llvm/llvm-project/pull/156355)**
*"[mlir][ptr] Add translations to LLVMIR for ptr ops."*
(merged 2025-09-03, author: `fabianmcg`)

This PR implemented `PtrToLLVMIRTranslation`, which translates `ptr.load` →
`llvm::LoadInst` (with full support for `volatile`, `atomic`,
`nontemporal`, `invariant`, and `syncscope`) **directly to LLVM IR**,
bypassing the LLVM dialect entirely.

The companion **PR [#156333](https://github.com/llvm/llvm-project/pull/156333)**
*"[mlir][LLVM|ptr] Add the `#llvm.address_space` attribute, and allow `ptr` translation"*
(merged 2025-09-02) added the `#llvm.address_space` attribute required to
make `!ptr.ptr` types translatable.

### 4. The design direction: prefer translation over dialect conversion

**PR [#157347](https://github.com/llvm/llvm-project/pull/157347)**
*"[mlir][ptr] Add ConstantOp with NullAttr and AddressAttr support"*
(merged 2025-09-14) also renamed `convert*` identifiers to `translate*` in
the `PtrToLLVM` file, explicitly signalling the intent to move away from
dialect conversion towards direct translation.

## Conclusion

The absence of a `ptr.load → LLVM::LoadOp` conversion is **by design, not
an oversight**:

1. When `ptr.load` was introduced (PR #156093), its LLVM lowering was
   explicitly deferred as "future work".
2. The team's preferred path is **direct LLVM IR translation** (via
   `PtrToLLVMIRTranslation`), rather than first lowering to LLVM dialect ops.
3. The existing `PtrToLLVM` dialect conversion infrastructure is described as
   a "stop gap" and covers only the older ptr ops, not load/store.

## Current Status (as of the research date)

| Path | Availability for `ptr.load` |
|------|-----------------------------|
| LLVM IR translation (`mlir-translate`) | ✅ Available (PR #156355, merged 2025-09-03) |
| LLVM dialect conversion (`mlir-opt --convert-to-llvm`) | ❌ Not available (future work per PR #156093) |

## Workaround

If you need `ptr.load → llvm.load` today, you can write a small
`ConvertOpToLLVMPattern<ptr::LoadOp>` that maps the op's attributes
(alignment, volatile, atomic ordering, syncscope, nontemporal, invariant)
to the corresponding `LLVM::LoadOp` attributes.  Alternatively, use the
`mlir-translate` path if your pipeline ends in LLVM IR.

## Where to Ask for Updates

- **LLVM Discourse (MLIR category)**: https://discourse.llvm.org/c/mlir/
  — best for design questions and "when is this landing?" discussions.
- **GitHub issue tracker** (`llvm/llvm-project`): file a feature request
  if you need this conversion and want upstream visibility.

## References

| PR | Title | Date |
|----|-------|------|
| [#73057](https://github.com/llvm/llvm-project/pull/73057) | [mlir] Ptr dialect (prototype) | 2023-11-22 |
| [#86860](https://github.com/llvm/llvm-project/pull/86860) | [mlir][Ptr] Init the Ptr dialect with the `!ptr.ptr` type | merged 2024-06-27 |
| [#86870](https://github.com/llvm/llvm-project/pull/86870) | [mlir][Ptr] Add the `MemorySpaceAttrInterface` interface | merged 2025-03-19 |
| [#156053](https://github.com/llvm/llvm-project/pull/156053) | [mlir][ptr] Add conversion to LLVM for all existing `ptr` ops | merged 2025-08-29 |
| [#156093](https://github.com/llvm/llvm-project/pull/156093) | [mlir][ptr] Add load and store ops | merged 2025-09-01 |
| [#156333](https://github.com/llvm/llvm-project/pull/156333) | [mlir][LLVM\|ptr] Add the `#llvm.address_space` attribute | merged 2025-09-02 |
| [#156355](https://github.com/llvm/llvm-project/pull/156355) | [mlir][ptr] Add translations to LLVMIR for ptr ops | merged 2025-09-03 |
| [#157347](https://github.com/llvm/llvm-project/pull/157347) | [mlir][ptr] Add ConstantOp with NullAttr and AddressAttr support | merged 2025-09-14 |
| [RFC on Discourse](https://discourse.llvm.org/t/rfc-ptr-dialect-modularizing-ptr-ops-in-the-llvm-dialect/75142) | [RFC] `ptr` dialect & modularizing ptr ops in the LLVM dialect | 2023 |
