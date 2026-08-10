# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
