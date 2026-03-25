---
title: 使用实验性质的算子
weight: 40
---

<!--
# Using Experimental Operators
-->
# 使用实验性质的算子

<!--
The `experimental_ops` module provides a space for new operators
that are not yet ready for production release.
Operators in this package are accessible via `flag_gems.experimental_ops.*`.
These operators follow the same development patterns as the core, stable operators.
-->
*FlagGems* 的 `experimental_ops` 模块提供了一个名字空间，
用来存放尚未为生产环境使用准备就绪的算子。
在这个包中的算子可以通过 `flag_gems.experimental_ops.*` 的形式来访问。
实验性质算子的开发与与核心的稳定算子相同的开发模式。

```python
from flag_gems import experimental_ops as ops

result = ops.rmsnorm(*args)
```

<!--
You can also use experimental operators in a `use_gems()` context,
however, you have to explicitly specify the full path for accessing the operator.
-->
你也可以在 `use_gems()` 所构造的上下文管理器中使用实验性质的算子，
不过你必须使用算子的完整包名才能访问到这类算子。

```python
with flag_gems.use_gems():
    result = flag_gems.experimental_ops.rmsnorm(*args)
```

<!--TODO(Qiming): Add link to experimental operators-->
