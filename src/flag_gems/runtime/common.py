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
    SPACEMIT = 13
    THEAD = 14
    ARM = 15

    @classmethod
    def get_all_vendors(cls) -> dict:
        vendorDict = {}
        for member in cls:
            vendorDict[member.name.lower()] = member
        return vendorDict


DEFAULT_STRATEGIES = {
    "addmm": ["align32", "align32", "align32"],
    "addmm_sqmma": ["align32", "align32", "align32"],
    "baddbmm": ["align32", "align32", "align32"],
    "bmm": ["align32", "align32", "align32", "align32", "align32"],
    "bmm_sqmma": ["align32", "align32", "align32"],
    "fused_marlin_moe_mxfp4": ["align32", "align32", "align32", "default"],
    "gemv": ["align32", "align32", "align32", "default"],
    "mm": ["align32", "align32", "align32", "align32", "align32"],
    "mm_sqmma": ["align32", "align32", "align32", "default"],
    "mm_general_tma": [
        "align32",
        "align32",
        "align32",
        "align32",
        "align32",
        "default",
    ],
    "mv": ["align32", "align32"],
    "sparse_attention": ["align32", "align32", "align32"],
    "w8a8_block_fp8_general": [
        "align32",
        "align32",
        "align32",
        "align32",
        "align32",
    ],
    "w8a8_block_fp8_general_splitk": [
        "align32",
        "align32",
        "align32",
        "align32",
        "align32",
    ],
    "w8a8_block_fp8_general_tma": [
        "align32",
        "align32",
        "align32",
        "align32",
        "align32",
        "default",
    ],
    "w8a8_block_fp8_bmm": ["default", "align32", "align32", "align32"],
    "mm_splitk": ["align32", "align32", "align32", "align32", "align32"],
}

OP_KEY_ORDERS = {
    "addmm": ["M", "N", "K"],
    "addmm_sqmma": ["M", "N", "K"],
    "bmm": ["M", "N", "K", "stride_am", "stride_bk"],
    "bmm_sqmma": ["M", "N", "K"],
    "baddbmm": ["M", "N", "K"],
    "fused_marlin_moe_mxfp4": ["N", "K", "BLOCK_SIZE_M", "SWAP_AB"],
    "gemv": ["M", "K", "stride_am", "stride_bk"],
    "mm": ["M", "N", "K", "stride_am", "stride_bk"],
    "mm_sqmma": ["M", "N", "K", "dtype"],
    "mm_general_tma": ["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    "mv": ["M", "N"],
    "sparse_attention": ["topk", "H_ACTUAL", "D"],
    "w8a8_block_fp8_general": ["M", "N", "K", "stride_am", "stride_bk"],
    "w8a8_block_fp8_general_splitk": ["M", "N", "K", "stride_am", "stride_bk"],
    "w8a8_block_fp8_general_tma": ["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    "w8a8_block_fp8_bmm": ["B", "M_aligned", "N", "K"],
    "mm_splitk": ["M", "N", "K", "stride_am", "stride_bk"],
}


# Mapping from vendor name to torch attribute for quick detection
_VENDOR_TORCH_ATTR = {
    "ascend": "npu",
    "cambricon": "mlu",
    "enflame": "gcu",
    "hygon": "__hcu_version__",
    "iluvatar": "corex",
    "mthreads": "musa",
    "sunrise": "ptpu",
}

__all__ = [
    "vendors",
    "DEFAULT_STRATEGIES",
    "OP_KEY_ORDERS",
    "_VENDOR_TORCH_ATTR",
]
