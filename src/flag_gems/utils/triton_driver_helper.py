try:
    from flag_gems.runtime import torch_device_fn

    get_device_properties = torch_device_fn.get_device_properties
except AttributeError:
    import triton

    get_device_properties = triton.runtime.driver.active.utils.get_device_properties
