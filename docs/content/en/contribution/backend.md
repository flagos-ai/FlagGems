---
title: Adding a new backend
weight: 40
---

# Adding a New Backend

## 1. Introduction

The `flag_gems` accelerated operators library from the *FlagGems* project can be used on multiple backends.
If you are a chip vendor and wish to contribute backend-specific optimizations for your hardware,
you can use this documentation to integrate the optimizations into *FlagGems*..

## 2. Create a backend directory

All vendor-specific optimization code reside in the `src/flag_gems/runtime/backend` directory.
You can start by creating a folder for identification under this directory,
following the naming pattern `<_vendor-name>`. As an example, all NVIDIA-specific customization
can be found at `src/flag_gems/runtime/backend/_nvidia`.

## 3. Initialize the directory

Create the necessary files, including but not limited to the `__init__.py` file,
the `heuristics_config_utils.py` file, the `tune_configs.yaml` file , as well as a folder named  `ops`.
The expected directory layout is shown in the following example:

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

### 3.1 About `__init__.py` file

An easy way to to create this file is to copy one from existing vendors
(say `src/flag_gems/runtime/backend/_nvidia/__init__.py`).
After having created your `__init__.py` file, the **only change** you need to make is
to configure the properties for the `VendorInfoBase` class:

```python
vendor_info = VendorInfoBase(
    vendor_name="<your vendor name>",
    device_name="<the device name>",
    device_query_cmd="<command for querying hardware info>"
)
```

The important properties for `VendorInfoBase` are:

- `vendor_name`: the vendor name at your choice, e.g. `nvidia`;
- `device_name`: the name for your acclerator device, e.g. `cuda`;
- `device_query_cmd`: the command line that is used to check the hardware devices
  on the node, e.g. `nvidia-smi`.
- `dispatch_key`: an optional property for registering operators to `torch.library.Library`
  in PyTorch, e.g. `PrivateUse1`.

### 3.2 The `heuristics_config_utils.py` file

In the `heuristics_config_utils.py` file, You will configure the `triton.heuristics` parameters.
You can  refer to `src/flag_gems/runtime/backend/_nvidia/heuristics_config_utils.py`
for an example.

### 3.3 The `tune_configs.yaml` file

In the `tune_configs.yaml` file, you can customize `triton.autotune` parameters.
Similarly, you can refer to `src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml`
for an example.

### 3.4 The `ops` directory

The `ops` directory is where vendor-customized operator implementations are stored.
For instance, if you want to create a custom `add` operator, you will place the implementation
in `ops/add.py`. Following that, you should update the `ops/__init__.py` accordingly
as shown in the following example. The `__all__` list in the `ops/__init__.py` file
ensures that your implementation for `add` and `gelu` is accessible from external
packages.

```python
from .add import add
from .gelu import gelu

__all__= ["add", "gelu"]
```
