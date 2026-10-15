---
title: Selecting Visible Devices
weight: 25
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

# Selecting Visible Devices

FlagGems runs on many accelerator backends, and each vendor restricts process-visible
devices through its own environment variable. There is **no vendor-neutral name** for
this variable, and FlagGems deliberately does not invent one: it reads the variable
owned by the active backend.

> **Warning** — every backend uses a *different* variable name. A name that works on one
> backend is silently ignored on another. Do not assume the variable you know from one
> vendor applies to the backend you are currently targeting.

## Background {#background}

Most accelerator runtimes follow the same contract, borrowed from `CUDA_VISIBLE_DEVICES`:

- The variable holds a comma-separated list of physical device indices, for example `0,2`.
- It must be set **before** the process starts — the runtime reads it once while
  initializing, so changing it at run time has no effect.
- The runtime **renumbers** the surviving devices. With `TOPS_VISIBLE_DEVICES=4,6`, the
  two GCUs that remain are visible to the process as logical devices `0` and `1`, *not*
  as `4` and `6`.

That last point is the one that most often causes confusion when a script does
`torch.gcu.set_device(4)` under a visibility filter: index `4` no longer exists.

## Supported backends {#supported-backends}

The table below lists the visibility variable for every backend known to FlagGems. It is
taken from `vendor_env_map` in
[`tools/run_tests.py`](https://github.com/flagos-ai/FlagGems/blob/master/tools/run_tests.py#L468-L495),
which is the authoritative list: the test driver reads it to hand a device subset to each
backend.

| Vendor     | Vendor name  | Variable                                           |
| ---------- | ------------ | -------------------------------------------------- |
| Ascend     | `ascend`     | `ASCEND_RT_VISIBLE_DEVICES`, `NPU_VISIBLE_DEVICES` |
| Cambricon  | `cambricon`  | `MLU_VISIBLE_DEVICES`                              |
| Enflame    | `enflame`    | `TOPS_VISIBLE_DEVICES`                             |
| Hygon      | `hygon`      | `HIP_VISIBLE_DEVICES`                              |
| Iluvatar   | `iluvatar`   | `ILUVATAR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES` |
| Kunlunxin  | `kunlunxin`  | `CUDA_VISIBLE_DEVICES`                             |
| MetaX      | `metax`      | `MACA_VISIBLE_DEVICES`                             |
| Mthreads   | `mthreads`   | `MUSA_VISIBLE_DEVICES`                             |
| NVIDIA     | `nvidia`     | `CUDA_VISIBLE_DEVICES`                             |
| Sunrise    | `sunrise`    | `TANG_VISIBLE_DEVICES`                             |
| Thead      | `thead`      | `CUDA_VISIBLE_DEVICES`                             |
| TsingMicro | `tsingmicro` | `TXDA_VISIBLE_DEVICES`                             |

Note that several vendors name their variable after the **runtime**, not after the vendor
itself — `TOPS_VISIBLE_DEVICES` for Enflame, `MACA_VISIBLE_DEVICES` for MetaX,
`HIP_VISIBLE_DEVICES` for Hygon, `TANG_VISIBLE_DEVICES` for Sunrise. Guessing the name
from the vendor name is therefore unreliable, and this table is the thing to check.

`nvidia` is deliberately absent from `vendor_env_map`: it is the default fallback, so an
unlisted vendor also receives `CUDA_VISIBLE_DEVICES`. That fallback is convenient for
NVIDIA itself but is easy to mistake for real support — some vendors accept the variable
without honoring it. When the driver prints
`No vendor-specific device masking for '<vendor>', falling back to CUDA_VISIBLE_DEVICES`,
treat it as a signal to add an entry rather than as confirmation that filtering worked.

Where a vendor lists two variables, the driver **sets all of them** to the same value, so
they stay consistent. That differs from FlagGems FlagTune, which reads the first non-empty
variable as the source of truth and clears the aliases — see
[Notes for backend contributors](#notes-for-backend-contributors).

> [!WARNING]
> **On Ascend, use `ASCEND_RT_VISIBLE_DEVICES` and not a look-alike name.** The runtime
> variable is not the obvious `ASCEND_VISIBLE_DEVICES`; a near-miss spelling is accepted
> silently and simply does not filter anything, leaving all NPUs visible.
> Note also that `ASCEND_DEVICE_ID` is a different thing again — it sets the *default*
> device inside the process rather than restricting which devices are visible, so it is
> not a substitute. `tools/run_tests.py` lists `NPU_VISIBLE_DEVICES` as a secondary
> Ascend name.

> [!WARNING]
> **On Enflame, do not use `ENFLAME_VISIBLE_DEVICES`.** That name matches the vendor
> naming that the other backends follow and therefore looks plausible, but it is not
> implemented anywhere in the GCU stack. Setting it is a silent no-op: the process still
> sees every GCU on the node. The variable that works is `TOPS_VISIBLE_DEVICES`, which
> comes from the TOPS/GCU runtime rather than from the Enflame name.

## Verifying the setting {#verifying-the-setting}

A visibility variable is only meaningful if it was set before the process started. To
confirm which devices a process actually sees, print the device count from within the
process:

```shell
# NVIDIA
CUDA_VISIBLE_DEVICES=0,1 python -c \
  "import torch; print(torch.cuda.device_count())"

# Ascend
ASCEND_RT_VISIBLE_DEVICES=0,1 python -c \
  "import torch, torch_npu; print(torch.npu.device_count())"

# Enflame
TOPS_VISIBLE_DEVICES=0,1 python -c \
  "import torch, torch_gcu; print(torch.gcu.device_count())"
```

Each command should print `2`. If it prints the full device count of the node instead,
the variable was misspelled, is the wrong variable for that backend, or was exported too
late in the process.

## Notes for backend contributors {#notes-for-backend-contributors}

When you add a new backend, register its visibility variable so that the test and
benchmark drivers can restrict devices correctly:

- Add an entry to the `vendor_env_map` dictionary in
  [`tools/run_tests.py`](https://github.com/flagos-ai/FlagGems/blob/master/tools/run_tests.py#L472-L484),
  keyed by the vendor name used by the backend's `VendorDescriptor` (see
  [Adding a new backend](/FlagGems/contribution/backend/)). Do not edit the fallback
  branch instead — `vendor_env_map` is the single place a new backend is registered.
- If the runtime accepts several aliases, list them all, most authoritative first.
  `get_env` sets every name in the list to the same value, so the aliases stay consistent
  and an existing launcher on either name keeps working.
- Do not reuse another vendor's variable as a fallback unless the runtime genuinely
  honors it. Falling back to `CUDA_VISIBLE_DEVICES` on a non-CUDA backend, for example,
  produces a setting that is accepted but has no effect.
