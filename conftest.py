#TODO: enflame locally(entire file)
import torch


def _fmt(_str):
    _str = _str.replace(' ', '')
    _str = _str.replace('(', '')
    _str = _str.replace(')', '')
    _str = _str.replace(',', 'x')
    return _str


def fmt_id(val):
    if isinstance(val, torch.dtype):
        return str(val)[6:]
    if isinstance(val, tuple):
        return _fmt('x'.join(map(str, val)))
    if isinstance(val, torch.Tensor):
        return _fmt(str(val.shape)[12:-2])
    return _fmt(str(val))


def pytest_generate_tests(metafunc):
    if not hasattr(metafunc.function, 'pytestmark'):
        return

    for i in range(len(metafunc.function.pytestmark)):
        if metafunc.function.pytestmark[i].name != 'parametrize':
            continue
        if metafunc.function.pytestmark[i].kwargs.get('ids') is not None:
            continue
        metafunc.function.pytestmark[i].kwargs['ids'] = fmt_id
