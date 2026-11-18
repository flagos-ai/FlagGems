---
title: 设置设备可见性
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

<!--
# Selecting Visible Devices

FlagGems runs on many accelerator backends, and each vendor restricts process-visible
devices through its own environment variable. There is **no vendor-neutral name** for
this variable, and FlagGems deliberately does not invent one: it reads the variable
owned by the active backend.
-->
# 设置设备可见性

FlagGems 可以运行在多种加速器后端上，而每一种硬件都通过自己的一套环境变量来控制进程可见的设备。这些变量**没有统一的跨厂商名称**，FlagGems 也不去自创一个名字，而是直接读取当前生效的后端所对应的环境变量。

<!--
> [!WARNING]
> Every backend uses a *different* variable name. A name that works on one
> backend is silently ignored on another. Do not assume the variable you know from one
> vendor applies to the backend you are currently targeting.
-->
> [!WARNING]
> 每个后端使用的变量名**各不相同**。在某个后端上生效的变量名，换到另一个
> 后端上会被直接忽略。请不要想当然地认为您在某个厂商那里熟悉的变量名同样适用于
> 您当前所使用的后端。

<!--
## Background

Most accelerator runtimes follow the same contract, borrowed from `CUDA_VISIBLE_DEVICES`:

- The variable holds a comma-separated list of physical device indices, for example `0,2`.
- It must be set **before** the process starts — the runtime reads it once while
  initializing, so changing it at run time has no effect.
- The runtime **renumbers** the surviving devices. With `TOPS_VISIBLE_DEVICES=4,6`, the
  two GCUs that remain are visible to the process as logical devices `0` and `1`, *not*
  as `4` and `6`.

That last point is the one that most often causes confusion when a script does
`torch.gcu.set_device(4)` under a visibility filter: index `4` no longer exists.
-->
## 背景知识

多数加速器运行时都遵循同一套约定，这套约定源自 `CUDA_VISIBLE_DEVICES`：

- 变量取值为以逗号分隔的物理设备编号列表，例如 `0,2`。
- 变量必须在进程启动**之前**设置好——运行时只在初始化时读取一次，进程运行期间再修改它是没有效果的。
- 运行时会对着剩余的可用设备**重新编号**。例如设置 `TOPS_VISIBLE_DEVICES=4,6` 之后，保留下来的两张 GCU 卡在进程内看到的是逻辑设备 `0` 和 `1`，而**不是** `4` 和 `6`。

最后一点最容易造成困惑：如果在设置了可见性的情况下执行 `torch.gcu.set_device(4)`，会发现编号 `4` 已经不存在了。

<!--
## Supported backends

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
variable as the source of truth and clears the aliases.
-->
## 各后端支持的变量

下表列出了 FlagGems 已知的全部后端及其可见性变量。该表来自 [`tools/run_tests.py`](https://github.com/flagos-ai/FlagGems/blob/master/tools/run_tests.py#L468-L495) 中的 `vendor_env_map`，并以它为准：测试驱动程序正是读取这个字典，把指定的设备子集下发给各个后端。

| 厂商       | 厂商名        | 环境变量                                               |
| -------- | ---------- | -------------------------------------------------- |
| Ascend   | `ascend`   | `ASCEND_RT_VISIBLE_DEVICES`、`NPU_VISIBLE_DEVICES`  |
| Cambricon | `cambricon` | `MLU_VISIBLE_DEVICES`                              |
| Enflame  | `enflame`  | `TOPS_VISIBLE_DEVICES`                             |
| Hygon    | `hygon`    | `HIP_VISIBLE_DEVICES`                              |
| Iluvatar | `iluvatar` | `ILUVATAR_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`  |
| Kunlunxin | `kunlunxin` | `CUDA_VISIBLE_DEVICES`                            |
| MetaX    | `metax`    | `MACA_VISIBLE_DEVICES`                             |
| Mthreads | `mthreads` | `MUSA_VISIBLE_DEVICES`                             |
| NVIDIA   | `nvidia`   | `CUDA_VISIBLE_DEVICES`                             |
| Sunrise  | `sunrise`  | `TANG_VISIBLE_DEVICES`                             |
| Thead    | `thead`    | `CUDA_VISIBLE_DEVICES`                             |
| TsingMicro | `tsingmicro` | `TXDA_VISIBLE_DEVICES`                           |

需要注意的是，有若干厂商的变量名取自**运行时**而非厂商名本身——Enflame 用的是 `TOPS_VISIBLE_DEVICES`，MetaX 用的是 `MACA_VISIBLE_DEVICES`，Hygon 用的是 `HIP_VISIBLE_DEVICES`，Sunrise 用的是 `TANG_VISIBLE_DEVICES`。因此，仅凭厂商名去猜测变量名并不可靠，应当以上表为准。

`nvidia` 之所以没有出现在 `vendor_env_map` 中，是因为它是默认兜底项：未登记的厂商同样会被赋予 `CUDA_VISIBLE_DEVICES`。这一兜底对 NVIDIA 本身很方便，但很容易被误当成真正的支持——有些厂商会接受该变量，却并不真正读取它。当驱动程序打印出 `No vendor-specific device masking for '<vendor>', falling back to CUDA_VISIBLE_DEVICES` 时，应把它理解为“需要补充一条登记”的信号，而不是“过滤已经生效”的证明。

当一个厂商列出了两个变量时，驱动程序会**把二者都设置为同一个值**，以保持一致。这与 FlagGems FlagTune 的做法不同：后者以第一个非空变量为准，并清掉其余别名。

<!--
> [!WARNING]
> **On Ascend, do not use `ASCEND_VISIBLE_DEVICES`.** The vendor's own documentation is
> inconsistent here: that name appears in a few Ascend packages and examples but is
> *not* read by the runtime, so it is ignored and all NPUs stay visible.
> `ASCEND_RT_VISIBLE_DEVICES` is the variable the runtime actually honors.
> `ASCEND_DEVICE_ID` is a different thing again — it selects the *default* device inside
> the process rather than restricting which devices are visible.
-->
> [!WARNING]
> **在 Ascend 上要使用 `ASCEND_RT_VISIBLE_DEVICES`，而不要用与之形似的名字。**
> 运行时读取的并不是那个看起来最直观的 `ASCEND_VISIBLE_DEVICES`；写成近似的名字不会
> 报错，但也起不到任何过滤作用，所有 NPU 依然可见。
> 另外，`ASCEND_DEVICE_ID` 又是另一回事——它设置的是进程内的*默认*设备，并不限制哪些
> 设备可见，因此不能拿它来代替。`tools/run_tests.py` 中把 `NPU_VISIBLE_DEVICES` 列为
> Ascend 的次要名称。

<!--
> [!WARNING]
> **On Enflame, do not use `ENFLAME_VISIBLE_DEVICES`.** That name matches the vendor
> naming that the other backends follow and therefore looks plausible, but it is not
> implemented anywhere in the GCU stack. Setting it is a silent no-op: the process still
> sees every GCU on the node. The variable that works is `TOPS_VISIBLE_DEVICES`, which
> comes from the TOPS/GCU runtime rather than from the Enflame name.
-->
> [!WARNING]
> **在 Enflame 上不要使用 `ENFLAME_VISIBLE_DEVICES`。** 这个名字与其他后端所遵循的
> 厂商命名方式一致，因此看上去很合理，但它在 GCU 软件栈中根本没有实现。设置它不会
> 报错，也不会起任何作用：进程仍然能看到节点上的全部 GCU。真正生效的变量是
> `TOPS_VISIBLE_DEVICES`，它来自 TOPS/GCU 运行时，而不是取自 Enflame 这个厂商名。

<!--
## Verifying the setting

A visibility variable is only meaningful if it was set before the process started. To
confirm which devices a process actually sees, print the device count from within the
process:
-->
## 验证设置是否生效

只有在进程启动之前设置的可见性变量才有意义。要确认进程究竟能看到哪些设备，最简单的方法是在进程内部打印设备数量：

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

<!--
Each command should print `2`. If it prints the full device count of the node instead,
the variable was misspelled, is the wrong variable for that backend, or was exported too
late in the process.
-->
每条命令都应该输出 `2`。如果输出的反而是节点上的全部设备数量，那就说明变量拼写有误、用错了后端对应的变量，或者导出得太晚。

<!--
## Notes for backend contributors

When you add a new backend, register its visibility variable so that the test and
benchmark drivers can restrict devices correctly:
-->
## 给后端贡献者的提示

在新增一种后端时，请记得注册它的可见性变量，这样测试和性能测试驱动程序才能正确地限制所使用的设备：

<!--
- Add an entry to the `vendor_env_map` dictionary in
  [`tools/run_tests.py`](https://github.com/flagos-ai/FlagGems/blob/master/tools/run_tests.py#L472-L484),
  keyed by the vendor name used by the backend's `VendorDescriptor` (see
  [Adding a new backend](backend.md)). Do not edit the fallback
  branch instead — `vendor_env_map` is the single place a new backend is registered.
- If the runtime accepts several aliases, list them all, most authoritative first.
  `get_env` sets every name in the list to the same value, so the aliases stay consistent
  and an existing launcher on either name keeps working.
- Do not reuse another vendor's variable as a fallback unless the runtime genuinely
  honors it. Falling back to `CUDA_VISIBLE_DEVICES` on a non-CUDA backend, for example,
  produces a setting that is accepted but has no effect.
-->
- 在 [`tools/run_tests.py`](https://github.com/flagos-ai/FlagGems/blob/master/tools/run_tests.py#L472-L484) 的 `vendor_env_map` 字典中新增一项，键名使用后端 `VendorDescriptor` 里声明的厂商名（参见[添加一种新的后端](backend.md)）。请勿改为修改兜底分支——`vendor_env_map` 才是登记新后端的唯一位置。
- 如果运行时接受多个别名，请把它们全部列出，并把最权威的放在最前面。`get_env` 会把列表中的每个名字都设置为同一个值，因此各别名之间保持一致，已有启动脚本无论使用哪个名字都能继续正常工作。
- 除非运行时确实读取，否则不要把其他厂商的变量当作兜底。例如在非 CUDA 后端上回退到 `CUDA_VISIBLE_DEVICES`，最终只会得到一个被接受但毫无效果的环境变量。
