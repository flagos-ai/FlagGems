from enum import Enum

autograd_str = "Autograd"


class Autograd(Enum):
    enable = True
    disable = False

    @classmethod
    def get_optional_value(cls):
        return [member.name for member in cls]


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
