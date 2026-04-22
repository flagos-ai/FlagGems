# 在 mini_test.py 顶部加这行，看编译器日志输出到哪里
import sys
sys.stderr = sys.stdout

import torch
from flag_gems.runtime.backend._kunlunxin.fused.topk_softmax import topk_softmax

device = "cuda"
gating = torch.tensor([[1.2558, -0.0384, -0.9200, 1.3843]], device=device, dtype=torch.float32)
weights = torch.empty(1, 2, device=device, dtype=torch.float32)
indices = torch.empty(1, 2, device=device, dtype=torch.int32)
token_expert = torch.empty(1, 2, device=device, dtype=torch.int32)

# 调用 kernel
topk_softmax(weights, indices, token_expert, gating, False)

# 检查 indices 是否为整数
print(f"indices raw: {indices}")
print(f"indices reinterpret as float: {indices.view(torch.float32)}")
print(f"token_expert raw: {token_expert}")
print(f"token_expert reinterpret as float: {token_expert.view(torch.float32)}")

