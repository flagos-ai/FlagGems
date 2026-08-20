---
title: Overview
weight: 10
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


# Performance Benchmarking Overview

*FlagGems* operators in general provides better or at least comparable performance
when compared to operators from the native PyTorch library.
We use the `triton.testing.do_bench` from the Triton project for benchmarking.
The kernel data obtained are shown in the following graph.

![Operator Speedup](/FlagGems/images/speedup-20251225.png)

The chart above shows the speedup of FlagGems compared with the PyTorch ATen library
in eager mode. The speedup is calculated by averaging the speedup on each shape,
representing the overall performance of the operator.

To ensure that the performance of any new operators are within an acceptable range,
we require all contributions to the operators' inventory provide performance data.
You can benchmark your new operators (and the existing ones) using the *benchmark*
framework in *FlagGems*.

Check [operator benchmark](/FlagGems/performance/benchmark/) for instructions
on benchmark testing your operators.
