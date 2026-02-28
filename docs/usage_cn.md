## 使用方法

FlagGems 支持两种常见的使用模式：对 PyTorch ATen 算子打补丁（推荐）和显式调用 FlagGems 算子。

### (1) 全局启用 FlagGems（对 ATen 算子打补丁）

执行 `flag_gems.enable()` 后，支持的 `torch.*` / `torch.nn.functional.*`
调用将会自动分发（Dispatch）到 FlagGems 的实现上。

```python
import torch
import flag_gems

flag_gems.enable()

x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
y = torch.mm(x, x)
```

如果你只想在某个作用域内（例如用于基准测试）使用 FlagGems，请使用上下文管理器：

```python
import torch
import flag_gems

with flag_gems.use_gems():
    x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
    y = torch.mm(x, x)
```

### (2) 显式调用 FlagGems 算子

你也可以绕过 PyTorch 的分发机制，直接从 `flag_gems.ops` 中调用算子，此时无需调用 `enable()`：

```python
import torch
from flag_gems import ops
import flag_gems

a = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
b = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
c = ops.mm(a, b)
```

若要了解更多详情和高级选项（例如禁用特定算子、运行时日志等），请参阅
[`how_to_use_flaggems`](./how_to_use_flaggems.md)。
