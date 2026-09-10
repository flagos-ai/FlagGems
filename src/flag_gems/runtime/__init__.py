from . import backend, common, error
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

config_loader = ConfigLoader()
device = DeviceDetector()

"""
The dependency order of the sub-directory is strict, and changing the order arbitrarily may cause errors.
"""

# torch_device_fn is like 'torch.cuda' object
backend.set_torch_backend_device_fn(device.vendor_name)
torch_device_fn = backend.gen_torch_device_object()

# torch_backend_device is like 'torch.backend.cuda' object
torch_backend_device = backend.get_torch_backend_device_fn()


class _NullGuard:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_null_guard = _NullGuard()


def _make_device_guard():
    """Enter the device context only when the launch is not already on that device.

    Every op body wraps its launches in `with torch_device_fn.device(x.device)`.
    Under paddle both `x.device` (~2us, it builds a Device out of the Place) and
    the context enter/exit (~6us, it converts that Device back into a Place twice)
    show up in the launch path, even though the whole thing is a no-op whenever the
    tensor already sits on the current device -- the normal case. Taking the tensor
    instead of its device lets us compare Places directly (~1.4us total) and skip
    the context. Anything not on the current device still goes through the real
    context, so device selection keeps working.
    """
    dev_ctx = torch_device_fn.device

    def _fallback_guard(x):
        # accepts a tensor or a device denotation, like the paddle path below
        return dev_ctx(getattr(x, "device", x))

    try:
        import paddle

        expected_place = paddle.framework._current_expected_place
        expected_place()
    except Exception:
        return _fallback_guard

    def _is_current(dev, cur):
        if dev is cur or dev == cur:
            return True
        # `x.device.index` / plain ints, and Device objects that print like the place
        if isinstance(dev, int):
            return cur.is_gpu_place() and cur.gpu_device_id() == dev
        return str(dev) == str(cur)

    def device_guard(x):
        cur = expected_place()
        place = getattr(x, "place", None)
        if place is not None:  # a tensor
            if place == cur:
                return _null_guard
            return dev_ctx(x.device)
        if _is_current(x, cur):  # a device denotation
            return _null_guard
        return dev_ctx(x)

    return device_guard


device_guard = _make_device_guard()


def get_tuned_config(op_name):
    return config_loader.get_tuned_config(op_name)


def get_heuristic_config(op_name):
    return config_loader.get_heuristics_config(op_name)


def replace_customized_ops(_globals):
    event = backend.BackendArchEvent()
    arch_specialization_operators = event.get_arch_ops() if event.has_arch else None
    backend_customization_operators = backend.get_current_device_extend_op(
        device.vendor_name
    )
    if device.vendor != common.vendors.NVIDIA:
        try:
            for fn_name, fn in backend_customization_operators:
                _globals[fn_name] = fn
        except RuntimeError as e:
            error.customized_op_replace_error(e)
    if arch_specialization_operators:
        try:
            for fn_name, fn in arch_specialization_operators:
                _globals[fn_name] = fn
        except RuntimeError as e:
            error.customized_op_replace_error(e)


__all__ = ["*"]
