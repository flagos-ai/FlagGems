## Multiple backend adaptations
### Introduction
The `flag_gems` operator library provides the ability to access multiple backends.  If you are a chip vendor and wish to integrate  `your flag_gems code` into our official main branch, you simply need to follow these steps to complete the process.

#### step 1:
Create a folder named after your vendor in the  `FlagGems/src/flag_gems/runtime/backend directory`, following the pattern `_vendorname`. For example, you can refer to the structure of  `FlagGems/src/flag_gems/runtime/backend/_nvidia`.

#### step 2:
Create the necessary files, including but not limited to `__init__.py`, `heuristics_config_utils.py`, `tune_configs.yaml`, as well as a folder named  `ops`. This is an example under _nvidia file:
```
├── __init__.py
├── heuristics_config_utils.py
├── ops
│   ├── __init__.py
│   ├── add.py
│   └── gelu.py
└── tune_configs.yaml
```

##### step 2.1  `__init__.py`

You can copy `FlagGems/src/flag_gems/runtime/backend/_nvidia/__init__.py` and the  ***only change***  you need to make is to configure the `VendorInfoBase` class
```
VendorInfoBase(
    vendor_name="xxx", device_name="xxx", device_query_cmd="xxx"
)
```
###### Necessary fields:
- `vendor_name` is your vendorname, like `nvidia`

- `device_name` is your devicename, like `cuda`

- `device_query_cmd` is a command that can only be successfully executed on your vendor's device, like `nvidia-smi`

###### Optional fields:
- `dispatch_key` is The operator registration field of torch.library.Library in pytorch, like `PrivateUse1`

##### step 2.2  `heuristics_config_utils.py`

You should configure  `triton.heuristics` params in `FlagGems/src/flag_gems/runtime/backend/_vendorname/heuristics_config_utils.py`and you can  refer to `FlagGems/src/flag_gems/runtime/backend/_nvidia/heuristics_config_utils.py`

##### step 2.3  `tune_configs.yaml`
You should configure  `triton.autotune` params in `FlagGems/src/flag_gems/runtime/backend/_vendorname/tune_configs.yaml` and you can refer to `FlagGems/src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml`

##### step 2.4  `ops`
The `ops` directory is where `vendor-customized operator` implementations are stored. For instance, if you want to create a custom `add operation`, you should place the implementation in `ops/add.py`. Following that, you should configure ops/__init__.py accordingly.
```python
from .add import add
from .gelu import gelu

__all__= ["add", "gelu"]
```
