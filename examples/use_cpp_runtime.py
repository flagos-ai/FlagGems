import threading

import torch

import flag_gems  # noqa: F401

x = torch.randn(10, device="cuda:0")
out = torch.ops.flag_gems.add_tensor(x, x)
print(out)

x = torch.randn(10, device="cuda:1")
out = torch.ops.flag_gems.add_tensor(x, x)
print(out)

x = torch.randn(10, device="cuda:2")
out = torch.ops.flag_gems.add_tensor(x, x)
print(out)


def f(x):
    print(torch.ops.flag_gems.add_tensor(x, x))


t = threading.Thread(target=f, args=(torch.randn(10, device="cuda:3"),))
t.start()
t.join()


# compile
def f(x, y):
    return torch.ops.flag_gems.add_tensor(x, y)


F = torch.compile(f)

x = torch.randn(2, 1, 3, device="cuda:1", requires_grad=True)
y = torch.randn(4, 1, device="cuda:1", requires_grad=True)
out = F(x, y)
ref = x + y
print(out)
print(ref)
