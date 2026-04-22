---
# This file contains a prepared question intended to be submitted to the
# llvm/llvm-project issue tracker (https://github.com/llvm/llvm-project/issues).
# Copy the text below the separator line and paste it as a new issue there.
# ---
# DO NOT use this template to open issues in this (flagos-ai/FlagGems) repository.
---

<!-- ============================================================
     Copy everything from here down and paste into a new issue at
     https://github.com/llvm/llvm-project/issues/new
     ============================================================ -->

## [Question] Why is there no obvious MLIR conversion from `ptr.load` (Ptr dialect) to `LLVM::LoadOp` (LLVM dialect)?

### Summary

There is a direct translation from `ptr.load` to LLVM IR (a raw `llvm::LoadInst`) implemented in
`mlir/lib/Target/LLVMIR/Dialect/Ptr/PtrToLLVMIRTranslation.cpp`.
However, when working within MLIR conversion pipelines (i.e., lowering through LLVM *dialect* ops before
reaching LLVM IR), it is not obvious how to lower `ptr.load` to `LLVM::LoadOp`.

Specifically:

- `mlir/lib/Conversion/PtrToLLVM/PtrToLLVM.cpp` registers a `PtrToLLVMDialectInterface` that hooks into
  the `ConvertToLLVM` framework, but there does not appear to be a concrete `RewritePattern` (e.g.,
  `ConvertOpToLLVMPattern<ptr::LoadOp>`) that rewrites `ptr.load` to `LLVM::LoadOp`.
- There is no standalone `-convert-ptr-to-llvm` pass listed in the official MLIR pass catalogue
  (https://mlir.llvm.org/docs/Passes/), unlike analogous passes such as `-convert-vector-to-llvm` or
  `-convert-async-to-llvm`.

### Questions

1. **Is this intentional (by design) or an omission?**
   - If by design: what is the rationale? Is it because the semantics of `ptr.load` cannot be fully
     captured by `LLVM::LoadOp` without additional target/data-layout information (alignment, address
     space, provenance, etc.)?
   - If an omission: are there plans to add such a conversion, or is a contribution welcome?

2. **What is the recommended lowering path?**
   For a program containing `ptr.load`, what is the correct pipeline to reach LLVM dialect and
   ultimately LLVM IR?
   For example:
   - Use `-convert-to-llvm` after calling `registerConvertPtrToLLVMInterface()`?
   - Go directly to LLVM IR via `mlir-translate` (bypassing LLVM dialect entirely)?
   - Some other intermediate step?

3. **Are there existing discussions, PRs, or issues documenting the rationale?**
   Pointers to any prior design decisions or RFC threads would be very helpful.

### Observation: direct LLVM IR translation does exist

`mlir/lib/Target/LLVMIR/Dialect/Ptr/PtrToLLVMIRTranslation.cpp` contains a `translateLoadOp` function
that produces a `llvm::LoadInst`, including handling of:
- alignment (`MaybeAlign`)
- `volatile`
- atomic ordering and sync scope
- nontemporal / invariant / invariant_group metadata

This shows the mapping from `ptr.load` to a real LLVM load is well-defined. The question is why a
parallel `ptr.load → LLVM::LoadOp` conversion pattern (for use within MLIR lowering pipelines) is
absent or not discoverable.

### Minimal reproducible example

#### Input (`ptr.load` in Ptr dialect)

```mlir
// example.mlir
func.func @load_example(%p: !ptr.ptr<#ptr.memory_space<0>>) -> i32 {
  %v = ptr.load %p : !ptr.ptr<#ptr.memory_space<0>> -> i32
  return %v : i32
}
```

#### Expected LLVM dialect form

```mlir
// After lowering to LLVM dialect (expected)
llvm.func @load_example(%p: !llvm.ptr) -> i32 {
  %v = llvm.load %p : !llvm.ptr -> i32
  return %v : i32
}
```

#### Steps tried

```bash
mlir-opt example.mlir \
  --convert-to-llvm \
  --mlir-print-ir-after-all 2>&1 | head -40
```

This either fails with a legalization error on `ptr.load`, or the op passes through unconverted,
depending on which interfaces are registered.

### Environment

- LLVM/MLIR version: `main` branch (as of `<insert git commit hash or YYYY-MM-DD date here>`)
- Relevant files inspected:
  - `mlir/include/mlir/Conversion/PtrToLLVM/PtrToLLVM.h`
  - `mlir/lib/Conversion/PtrToLLVM/PtrToLLVM.cpp`
  - `mlir/lib/Target/LLVMIR/Dialect/Ptr/PtrToLLVMIRTranslation.cpp`
  - `mlir/include/mlir/Dialect/Ptr/IR/PtrOps.td`

### References

- Ptr dialect documentation: https://mlir.llvm.org/docs/Dialects/PtrOps/
- RFC – "Ptr dialect: Modularizing ptr ops in the LLVM dialect":
  https://discourse.llvm.org/t/rfc-ptr-dialect-modularizing-ptr-ops-in-the-llvm-dialect/75142
- MLIR Passes documentation (no `convert-ptr-to-llvm` listed):
  https://mlir.llvm.org/docs/Passes/
- `PtrToLLVM.h` Doxygen: https://mlir.llvm.org/doxygen/PtrToLLVM_8h.html
