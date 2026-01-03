## Usage

FlagGems supports two common usage patterns: patching PyTorch ATen ops (recommended) and calling FlagGems ops explicitly.

### (1) Enable FlagGems globally (patch ATen ops)

After `flag_gems.enable()`, supported `torch.*` / `torch.nn.functional.*` calls will be dispatched to FlagGems implementations automatically.

```python
import torch
import flag_gems

flag_gems.enable()

x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
y = torch.mm(x, x)
```

If you only want FlagGems inside a scope (e.g., for benchmarking), use the context manager:

```python
import torch
import flag_gems

with flag_gems.use_gems():
    x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
    y = torch.mm(x, x)
```

### (2) Explicitly call FlagGems ops

You can also bypass PyTorch dispatch and call operators from `flag_gems.ops` directly (no `enable()` required):

```python
import torch
from flag_gems import ops
import flag_gems

a = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
b = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
c = ops.mm(a, b)
```

For more details and advanced options (disabling specific ops, runtime logging,e.g.), see
[`how_to_use_flaggems`](docs/how_to_use_flaggems.md).
