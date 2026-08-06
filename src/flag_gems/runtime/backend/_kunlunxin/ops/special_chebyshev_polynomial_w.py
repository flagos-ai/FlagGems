# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch


_CPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)


def special_chebyshev_polynomial_w(x, n):
    x_cpu = x.to(device="cpu")
    n_cpu = (
        n.to(device="cpu")
        if isinstance(n, torch.Tensor)
        else torch.tensor(n, dtype=torch.int64)
    )
    result = torch.ops.aten.special_chebyshev_polynomial_w.default.redispatch(
        _CPU_KEYSET, x_cpu, n_cpu
    )
    return result.to(device=x.device, dtype=x.dtype)


def special_chebyshev_polynomial_w_out(x, n, out):
    result = special_chebyshev_polynomial_w(x, n)
    out.copy_(result)
    return out
