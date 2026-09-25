---
title: HyV4 MUSA Operator Validation
weight: 40
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# HyV4 MUSA Operator Validation

This page defines the evidence required before a FlagGems implementation is
claimed as covered by a HyV4 MUSA inference path. A model-level blacklist or
fallback is an observation, not proof that the corresponding FlagGems operator
is incorrect.

## Operator families

HyV4 exercises the following high-priority families:

| Family | Representative input contract | Validation focus |
|---|---|---|
| MXFP8 MoE | hidden 6144, 256 experts, top-k 8, intermediate 2048, group size 128 | scale semantics, quantization, grouped GEMM, routing and accumulation dtype |
| DSA prefill | paged KV, index top-k 2048 | sparse attention accuracy and page-table transforms |
| DSA KV gather/dequant | paged FP8 KV, selected indices `[T, 2048]`, BF16 output | index bounds, scale layout and output accuracy |
| DSA top-k | logits `[T, S]`, k=2048, int64 indices | values, indices and deterministic tie behavior |
| mHC post | input `[T, 6144]`, residual `[T, 4, 6144]`, post `[T, 4]` | broadcast order and FP32 accumulation |
| clamped SwiGLU | gate/up `[T, 4096]`, output `[T, 2048]` | clamp, sigmoid and output dtype |
| non-contiguous BMM | model-derived shapes and strides | stride correctness without forcing a contiguous copy |
| repeat/index/copy/reduction | dynamic token, expert and page metadata | dispatch ownership, dynamic shapes and integer bounds |

The exact token dimension `T`, strides, dtypes and scale tensors must be
captured from the effective model runtime. The representative dimensions above
must not be converted into a synthetic claim that all dynamic shapes work.

## Required issue and pull-request evidence

Every independently fixable operator issue should include:

1. the full FlagGems, compiler, PyTorch and device-stack commits;
2. a saved real-shape input or a deterministic generator with a checksum;
3. the actual dispatch owner and kernel name;
4. expected and actual outputs with maximum and mean error;
5. eager and compiled-path coverage;
6. a test that fails before the fix and passes after it;
7. operator latency and model-level A/B results;
8. regression coverage for unaffected backends;
9. a rollback description.

If the effective owner is PyTorch-MUSA, the compiler backend, SGLang or the
communication library, route the issue to that project instead of adding a
FlagGems workaround.

## Safe selective enablement

When model integration uses selective operator enablement, retain the runtime
dispatch record. An empty record must be reported explicitly; it commonly
means an existing PrivateUse1 implementation owned the call and FlagGems
skipped duplicate registration.

Do not claim FlagGems model coverage from any of the following alone:

- the operator exists in the source tree;
- the operator name appears in an allowlist or blacklist;
- the service returns HTTP 200;
- a model benchmark passes while the FlagGems dispatch record is empty.

## End-to-end regression gate

After operator-level validation, rerun the same model fingerprint with:

- bounded deterministic generation;
- repeated and concurrent requests;
- the selected accuracy workload;
- profiler-disabled performance measurements;
- worker, scheduler, OOM, NaN and device-error checks.

Any operator, precision, dispatch, graph or cache change invalidates older
model-level evidence until these gates are repeated.

