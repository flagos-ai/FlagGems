from enum import Enum


class vendors(Enum):
    NVIDIA = 0
    CAMBRICON = 1
    METAX = 2
    ILUVATAR = 3
    MTHREADS = 4
    KUNLUNXIN = 5
    HYGON = 6
    AMD = 7
    AIPU = 8
    ASCEND = 9
    TSINGMICRO = 10
    SUNRISE = 11
    ENFLAME = 12

    @classmethod
    def get_all_vendors(cls) -> dict:
        vendorDict = {}
        for member in cls:
            vendorDict[member.name.lower()] = member
        return vendorDict


UNSUPPORT_FP64 = frozenset(
    {
        vendors.CAMBRICON,
        vendors.ILUVATAR,
        vendors.KUNLUNXIN,
        vendors.MTHREADS,
        vendors.AIPU,
        vendors.ASCEND,
        vendors.TSINGMICRO,
        vendors.SUNRISE,
        vendors.ENFLAME,
    }
)

UNSUPPORT_BF16 = frozenset(
    {
        vendors.AIPU,
        vendors.SUNRISE,
    }
)

UNSUPPORT_INT64 = frozenset(
    {
        vendors.AIPU,
        vendors.TSINGMICRO,
        vendors.SUNRISE,
        vendors.ENFLAME,
    }
)


# Mapping from vendor name to torch attribute for quick detection
_VENDOR_TORCH_ATTR = {
    "cambricon": "mlu",
    "mthreads": "musa",
    "iluvatar": "corex",
    "ascend": "npu",
    "sunrise": "ptpu",
    "enflame": "gcu",
}
