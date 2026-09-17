---
title: Using Experimental Operators
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

# Using Experimental Operators

The `experimental_ops` module provides a space for new operators
that are not yet ready for production release.
Operators in this package are accessible via `flag_gems.experimental_ops.*`.
These operators follow the same development patterns as the core, stable operators.

```python
from flag_gems import experimental_ops as ops

result = ops.rmsnorm(*args)
```

You can also use experimental operators in a `use_gems()` context,
however, you have to explicitly specify the full path for accessing the operator.

```python
with flag_gems.use_gems():
    result = flag_gems.experimental_ops.rmsnorm(*args)
```

<!--TODO(Qiming): Add link to experimental operators-->
