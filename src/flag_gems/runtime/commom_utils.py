from enum import Enum

from .backend import backend_utils

Autograd = backend_utils.Autograd


class vendors(Enum):
    NVIDIA = 0
    CAMBRICON = 1
    METAX = 2
    ILUVATAR = 3
    MTHREADS = 4
    KUNLUNXIN = 5
    HYGON = 6
    AMD = 7

    @classmethod
    def get_all_vendors(cls):
        return [member.name for member in cls]


vendors_map = {
    "nvidia": vendors.NVIDIA,
    "cambricon": vendors.CAMBRICON,
    "iluvatar": vendors.ILUVATAR,
    "kunlunxin": vendors.KUNLUNXIN,
    "mthreads": vendors.MTHREADS,
    "hygon": vendors.HYGON,
    "metax": vendors.METAX,
    "AMD": vendors.AMD,
}
