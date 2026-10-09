---
title: Backend Optimization
weight: 45
bookCollapseSection: true
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

# Backend optimization

Triton launch options and useful tile sizes depend on the device and compiler. Start with the architecture and compiler notes for your backend, then measure representative shapes with the [FlagGems tuning workflow](/FlagGems/usage/tuning/).

- [Ascend](ascend/): AI Core memory and compute paths, Triton-Ascend options, and NPU tuning.
- [Hygon](hygon/): DCU/HIP memory hierarchy, Triton launch options, and tile tuning.

Treat the configurations in these guides as candidates. Check numerical results and latency on the target device and installed compiler before adopting them.
