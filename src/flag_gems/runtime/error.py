def backend_not_support(device_name, backend_list):
    raise RuntimeError(f"The {device_name} device is not supported currently. ")


def device_not_found():
    raise RuntimeError(
        "No device were detected on your machine ! \n "
        "Please check that your driver is complete. "
    )


def register_error(e):
    raise RuntimeError(
        e, "An error was encountered while registering the triton operator."
    )
