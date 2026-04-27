---
title: 添加一种新的后端
weight: 40
---

<!--
# Adding a New Backend

## 1. Introduction
-->
# 添加一种新的后端

## 1. 介绍

<!--
The `flag_gems` accelerated operators library from the *FlagGems* project can be used on multiple backends.
If you are a chip vendor and wish to contribute backend-specific optimizations for your hardware,
you can use this documentation to integrate the optimizations into *FlagGems*..
-->
*FlagGems* 项目所提供的 `flag_gems` 加速算子库可以用在多种不同的后端平台上。
如果你是一个芯片厂商并且贡献一些针对自己的硬件做了特定于后端的优化，
你可以使用本文档来将自己所做的优化集成到 *FlagGems* 中。

<!--
## Create a backend directory

All vendor-specific optimization code reside in the `src/flag_gems/runtime/backend` directory.
You can start by creating a folder for identification under this directory,
following the naming pattern `<_vendor-name>`. As an example, all NVIDIA-specific customization
can be found at `src/flag_gems/runtime/backend/_nvidia`.
-->
## 2. 创建后端目录

所有针对具体厂商作了优化的代码都保存在 `src/flag_gems/runtime/backend` 目录之下。
作为起始的第一步，你可以在这个目录下面创建一个新的文件夹，用于区分自己与其他厂商。
请遵守目录的命名约定，格式为 `<_厂商名>`。
例如，所有针对 NVIDIA 所定制的代码都位于目录 `src/flag_gems/runtime/backend/_nvidia`
下面。

<!--
## 3. Initialize the directory

Create the necessary files, including but not limited to the `__init__.py` file,
the `heuristics_config_utils.py` file, the `tune_configs.yaml` file , as well as a folder named  `ops`.
The expected directory layout is shown in the following example:
-->
## 3. 初始化后端目录

你需要在新建的后端目录下创建一些必要的文件，包括但不限于 `__init__.py` 文件、
`heuristics_config_utils.py` 文件、`tune_configs.yaml` 文件；
此外还需要创建一个名为 `ops` 的文件夹。
下面的例子展示了期望的目录结构布局：

```none
├── __init__.py
├── heuristics_config_utils.py
├── ops
│   ├── __init__.py
│   ├── add.py
│   └── gelu.py
│   `── (other operators ...)
└── tune_configs.yaml
```

<!--
### 3.1 About `__init__.py` file

An easy way to to create this file is to copy one from existing vendors
(say `src/flag_gems/runtime/backend/_nvidia/__init__.py`).
After having created your `__init__.py` file, the **only change** you need to make is
to configure the properties for the `VendorInfoBase` class:
-->
### 3.1 关于 `__init__.py` 文件

创建 `__init__.py` 文件的一种简单方式是从现有的厂商目录下复制一份
（比如 `src/flag_gems/runtime/backend/_nvidia/__init__.py`）。
创建了 `__init__.py` 文件之后，你所需要执行的**唯一更改**是配置
`VendorInfoBase` 类的属性。

```python
vendor_info = VendorInfoBase(
    vendor_name="<你要使用的厂商名称>",
    device_name="<设备名称>",
    device_query_cmd="<用来查询硬件信息的命令>"
)
```

<!--
The important properties for `VendorInfoBase` are:

- `vendor_name`: the vendor name at your choice, e.g. `nvidia`;
- `device_name`: the name for your acclerator device, e.g. `cuda`;
- `device_query_cmd`: the command line that is used to check the hardware devices
  on the node, e.g. `nvidia-smi`.
- `dispatch_key`: an optional property for registering operators to `torch.library.Library`
  in PyTorch, e.g. `PrivateUse1`.
-->
`VendorInfoBase` 类的主要属性如下：

- `vendor_name`：按自己需要选择的厂商名字，例如 `nvidia`；
- `device_name`：你的加速器设备的名称，例如 `cuda`；
- `device_query_cmd`：用来检查节点上硬件设备状态的命令，例如 `nvidia-smi`；
- `dispatch_key`：这是一个可选的属性，用来将算子注册到 PyTorch 框架中的
  `torch.library.Library`，例如 `PrivateUse1`。

<!--
### 3.2 The `heuristics_config_utils.py` file

In the `heuristics_config_utils.py` file, You will configure the `triton.heuristics` parameters.
You can  refer to `src/flag_gems/runtime/backend/_nvidia/heuristics_config_utils.py`
for an example.
-->
### 3.2 `heuristics_config_utils.py` 文件

你可以在 `heuristics_config_utils.py` 文件中配置 `triton.heuristics` 相关参数。
你可以参照 `src/flag_gems/runtime/backend/_nvidia/heuristics_config_utils.py`
的内容根据自己的设备特点定制参数。

<!--
### 3.3 The `tune_configs.yaml` file

In the `tune_configs.yaml` file, you can customize `triton.autotune` parameters.
Similarly, you can refer to `src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml`
for an example.
-->
### 3.3 `tune_configs.yaml` 文件

在 `tune_configs.yaml` 文件中，你可以定制 `triton.autotune` 相关的参数。
不出意外，你可以参照 `src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml`
的内容完成配置。

<!--
### 3.4 The `ops` directory

The `ops` directory is where vendor-customized operator implementations are stored.
For instance, if you want to create a custom `add` operator, you will place the implementation
in `ops/add.py`. Following that, you should update the `ops/__init__.py` accordingly
as shown in the following example. The `__all__` list in the `ops/__init__.py` file
ensures that your implementation for `add` and `gelu` is accessible from external
packages.
-->
### 3.4 `ops` 目录

目录 `ops` 用来存放厂商定制的算子实现代码。
例如，如果你希望为自己的设备创建一个定制的 `add` 算子，你应该将算子的实现代码放在
`ops/add.py` 文件中。创建这个文件之后，你还需要相应更新 `ops/__init__.py` 文件
（如下例所展示的那样）。`ops/__init__.py` 文件中的 `__all__` 列表可以确保你为
`add` 和 `gelu` 所给出的实现能够从外部的包中访问到。

```python
from .add import add
from .gelu import gelu

__all__= ["add", "gelu"]
```
