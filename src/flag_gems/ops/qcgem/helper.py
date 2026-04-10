# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

import torch
import gc
import time
from tqdm import tqdm

from .core import QCGeMLinear, DType, TORCH_TO_DTYPE
QCGemLinearTriton = QCGeMLinear  # Alias for QCGemV2.0 API compatibility

from .utils import M_MAPPING
from .quant_utils import WeightQuantizerMXFP
from .utils import IS_HIP

if IS_HIP:
    default_fp8 = torch.float8_e4m3fnuz #AMD
    #default_fp8 = torch.float8_e5m2fnuz #AMD
    #default_fp8 = torch.float8_e5m2 #AMD - fp16 emulated
    default_post_scale = True
else:
    default_fp8 = torch.float8_e4m3fn #Nvidia 
    #default_fp8 = torch.float8_e5m2 #Nvidia
    default_post_scale = False

#################################################################################################
#Clean-up
def cleanup_linear(linear_layer, del_orig=True):
    if del_orig:
        for attr in ['weight', 'bias', 'weight_scale', 'W_q', 'meta']:
            val = getattr(linear_layer, attr, None)
            if val is not None and hasattr(val, '__len__') and len(val) > 0:
                setattr(linear_layer, attr, None)
    torch.cuda.empty_cache()

#Replaces all linear layers with the corresponding processor
def patch_model(model, device, processor, skip_modules=['lm_head', 'vision', 'visual'], group_size=64):
    """
    Helper function to quantize the whole model from cpu device. 
    The group_size parameter is only used in HQQ for W_nbits <=4, all the other configs do not use this parameter.
    """
    if(hasattr(processor, "from_hqqlinear")):
        try:
            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        except ImportException:
            print("This processor requires the `hqq` package. Install it via `pip install hqq`.")
    else:
        class _NoHQQ: pass
        HQQLinear = _NoHQQ

    #Name modules
    for name, module in model.named_modules():
        module.name = name

    #Patching fct
    def _patching_fct(layer, device, skip_modules):
        layer = layer.to(device=device, non_blocking=True)
        if(any(s in layer.name for s in skip_modules)):
            return layer
        else:
            if(hasattr(processor, "from_hqqlinear")):
                if(isinstance(layer, torch.nn.Linear)):
                    W_nbits = processor.W_nbits
                    quant_config = BaseQuantizeConfig(nbits=W_nbits, group_size=group_size if W_nbits <=4 else None)
                    layer = HQQLinear(layer, quant_config=quant_config, compute_dtype=layer.weight.dtype, device=device)
                    layer = processor.from_hqqlinear(layer)
            else:
                layer = processor.from_linear(layer)

        return layer

    #Replaces linear layers
    def _patch_linearlayers(model, fct, device, skip_modules):
        for name, layer in model.named_children():
            if isinstance(layer, (torch.nn.Linear, HQQLinear)):
                setattr(model, name, fct(layer, device, skip_modules))
            else:
                _patch_linearlayers(layer, fct, device, skip_modules)

    #Apply patch
    _patch_linearlayers(model, _patching_fct, device, skip_modules)

    #Clean-up
    model = model.to(device=device)
    torch.cuda.empty_cache()
    gc.collect()

    return model

#16-bit activations / 8-bit weigths
class A16W8: #INT8/FP8 weight-only channel-wise
    def __init__(self, device='cuda:0', dtype=None, fp8=None, fp32_scale=True, post_scale=False):
        self.device = device
        self.dtype = dtype
        self.fp8 = fp8
        self.fp32_scale = fp32_scale
        self.post_scale = post_scale

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]

        if(scales is None):
            #Quantize
            if(self.fp8): #FP8
                w_dtype, min_val, max_val = self.fp8, torch.finfo(self.fp8).min, torch.finfo(self.fp8).max
            else: #INT8
                w_dtype, min_val, max_val = torch.int8, torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max

            dtype = weight.dtype if(self.dtype is None) else self.dtype
            
            assert dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ], f"Invalid weight dtype, should be floating point, got {dtype}"

            qcgem_dtype = TORCH_TO_DTYPE[dtype]
            data_ptr = weight.data_ptr()
            weight = weight.to(dtype=torch.float32, copy=False, device=self.device)
            scales = weight.abs().amax(axis=1, keepdim=True) / max_val
            scales.clamp_(min=1e-6)
            W_q = (weight / scales).clamp_(min=min_val, max=max_val)
            if(w_dtype.is_floating_point):
                W_q = W_q.to(w_dtype)
            else:
                W_q = W_q.round_().to(w_dtype)
            if(data_ptr != weight.data_ptr()):
                del weight
                torch.cuda.empty_cache()
        else:
            #Pre-Quantized
            assert weight.itemsize == 1, f"Invalid weight.dtype, should be 8-bit (INT8 or FP8), got {weight.dtype}"
            if(self.dtype is None):
                dtype = scales.dtype if scales.dtype in [torch.float16, torch.bfloat16] else torch.float16
            else:
                dtype = self.dtype
            W_q = weight.to(device=self.device)
            scales = scales.to(device=self.device)
            qcgem_dtype = TORCH_TO_DTYPE[dtype]

        #scales = scales.to(dtype=torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        qcgem_linear = QCGemLinearTriton(8, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=qcgem_dtype, 
                        output_dtype=qcgem_dtype, 
                        )

        qcgem_linear.pack(W_q, scales, zeros=None, bias=bias)

        if(self.post_scale):
            #Post-scale
            qcgem_linear.W_group_mode       = 0
            qcgem_linear.channel_scale_mode = 1
        else:
            #Pre-scaling
            qcgem_linear.W_group_mode       = 2
            qcgem_linear.channel_scale_mode = 0

        return qcgem_linear

    def from_linear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(weight=linear_layer.weight, bias=linear_layer.bias)
        cleanup_linear(linear_layer, del_orig)
        return out_layer

#FP16 activations / Wn packed weights
class A16Wn:  # 8/4/2-bit weight-only as grouped "INT" / 8/4-bit as MXFP type
    def __init__(
        self,
        device="cuda:0",
        dtype=None,
        packing_bitwidth=None,
        post_scale=default_post_scale,
    ):
        self.post_scale = post_scale
        self.device = device
        self.dtype = dtype
        self.packing_bitwidth = packing_bitwidth
        self.quantizer_mx = None
        self.mx_fp8_dtype = default_fp8

    def from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias=None, quant_type = "INT"):
        return self.from_weights_(W_q, scales, zeros, W_nbits, group_size, bias, quant_type)

    def from_weights_(self, W_q, scales, zeros, W_nbits, group_size, bias=None, quant_type = "INT"):
        assert quant_type in ["INT", "MXFP"], f"Invalid quant_type. Got {quant_type}, valid values are INT, MXFP."

        if(quant_type == "MXFP"):
            assert W_nbits in [8, 4], "Unsupported W_nbit for MXFP quant_dtype."
            assert group_size == 32, "group_size should 32 for MXFP."

        if(isinstance(W_q, torch.nn.Parameter)):
            W_q = W_q.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if(quant_type == "INT"):
            dtype = scales.dtype if(self.dtype is None) else self.dtype
            scales = scales.to(dtype)

            assert scales.dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ], "Invalid scales.dtype, should floating point."
            
            qcgem_dtype = TORCH_TO_DTYPE[dtype]
            scales = scales.to(dtype)
            zeros = zeros.to(dtype)

        elif(quant_type == "MXFP"):
            if(W_nbits == 8):
                assert W_q.dtype in [self.mx_fp8_dtype], f"Unsupported dtype of W_q. got {W_q.dtype}"
            if(W_nbits == 4):
                assert W_q.dtype in [torch.uint8], f"Unsupported dtype of W_q. got {W_q.dtype}"

            dtype = torch.float16 if (self.dtype is None) else self.dtype
            if(dtype == torch.float16):
                qcgem_dtype = DType.MXFP16
            elif(dtype == torch.bfloat16):
                qcgem_dtype = DType.MXBF16
            else:
                raise Exception(f"Unsupported dtype for MXFP. Got {dtype}, supported [torch.float16, torch.bfloat16]")
            self.post_scale = False

            N, K = W_q.shape
            W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)#.float()

        in_features, out_features = W_q.shape[::-1] 

        W_q = W_q.to(self.device)
        scales = scales.to(device=self.device) if (scales is not None) else None
        zeros = zeros.to(device=self.device) if (zeros is not None) else None
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        qcgem_linear = QCGemLinearTriton(W_nbits, 
                        group_size=group_size,  
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=qcgem_dtype, 
                        output_dtype=qcgem_dtype, 
                        scaled_activations=False,
                        )

        qcgem_linear.pack(
            W_q, scales, zeros, bias=bias, packing_bitwidth=self.packing_bitwidth
        )

        if(group_size == in_features and quant_type == "INT"):
            if(self.post_scale):
                qcgem_linear.W_group_mode = 1
                qcgem_linear.channel_scale_mode = 1
            else:
                qcgem_linear.W_group_mode = 3
                qcgem_linear.channel_scale_mode = 0

        return qcgem_linear

    def from_hqqlinear(self, hqq_layer, del_orig=True):
        assert hqq_layer.meta['axis'] == 1, 'Only axis==1 is supported.'

        self.device = hqq_layer.W_q.device

        W_nbits    = hqq_layer.meta['nbits']
        group_size = hqq_layer.meta["group_size"]
        if(group_size is None):
            group_size = hqq_layer.in_features

        #Expects uint8 for Wn quantization!
        W_q    = hqq_layer.unpack(dtype=torch.uint8).view(hqq_layer.meta['shape'])
        scales = hqq_layer.meta['scale'].clone()
        zeros  = hqq_layer.meta['zero'].clone()
        bias   = hqq_layer.bias.clone() if (hqq_layer.bias is not None) else None  

        cleanup_linear(hqq_layer, del_orig)

        out_layer = self.from_weights_(
            W_q=W_q,
            scales=scales,
            zeros=zeros,
            W_nbits=W_nbits,
            group_size=group_size,
            bias=bias,
            quant_type="INT",
        )
        
        #Clean-up
        del W_q
        torch.cuda.empty_cache()

        return out_layer

    def mxfp_from_linear(self, linear_layer, W_nbits, del_orig=True):
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        group_size = 32
        N, K = W.shape
        if(W_nbits == 8):
            W_q, scales = self.quantizer_mx.quantize_mxfp8(W, index=True, mx_fp8_dtype=self.mx_fp8_dtype)
        if(W_nbits == 4):
            W_q, scales = self.quantizer_mx.quantize_mxfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)
        
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights_(
            W_q=W_q,
            scales=scales,
            zeros=None,
            W_nbits=W_nbits,
            group_size=group_size,
            bias=bias,
            quant_type="MXFP",
        )

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

#Alias
class A16W8_INT8(A16W8):
    def __init__(self, device='cuda:0', dtype=None, fp32_scale=True, post_scale=False):
        super().__init__(device=device, dtype=dtype, fp8=None, fp32_scale=fp32_scale, post_scale=post_scale)

class A16W8_FP8(A16W8):
    def __init__(self, device='cuda:0', dtype=None, fp8=default_fp8, fp32_scale=True, post_scale=False):
        super().__init__(device=device, dtype=dtype, fp8=fp8, fp32_scale=fp32_scale, post_scale=post_scale)

class A16Wn_HQQ_INT(A16Wn):
    def __init__(self, device='cuda:0', dtype=None, W_nbits=None):
        super().__init__(device=device, dtype=dtype)
        self.W_nbits = W_nbits

    def from_weights(self, W_q, scales, zeros, bias=None):
        group_size = W_q.numel() // scales.numel()
        return super().from_weights(
            W_q=W_q,
            scales=scales,
            zeros=zeros,
            W_nbits=self.W_nbits,
            group_size=group_size,
            bias=bias,
            quant_type="INT",
        )

class A16W8_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=8)

class A16W4_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=4)

class A16W2_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=2)

class A16W1_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=1)

class A16Wn_MXFP(A16Wn):
    def __init__(self, device='cuda:0', dtype=None, W_nbits=None):
        super().__init__(device=device, dtype=dtype)
        self.W_nbits = W_nbits

    def from_weights(self, W_q, scales, bias=None):
        group_size = W_q.numel() // scales.numel()
        return super().from_weights(
            W_q=W_q,
            scales=scales,
            zeros=None,
            W_nbits=self.W_nbits,
            group_size=group_size,
            bias=bias,
            quant_type="MXFP",
        )

    def from_linear(self, linear_layer, del_orig=True):
        return super().mxfp_from_linear(
            linear_layer=linear_layer, W_nbits=self.W_nbits, del_orig=del_orig
        )

class A16W8_MXFP(A16Wn_MXFP):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=8)

class A16W4_MXFP(A16Wn_MXFP):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=4)

#################################################################################################
#8-bit dynamic activations / 8-bit weights
#################################################################################################
class A8W8_dynamic:
    def __init__(self, device='cuda:0', dtype=None, fp8=False, fp32_scale=True):
        self.device = device
        self.dtype = dtype
        self.fp8 = fp8
        self.fp32_scale = fp32_scale

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if(self.fp8): #FP8
            w_dtype, input_dtype, min_val, max_val = self.fp8, TORCH_TO_DTYPE[self.fp8], torch.finfo(self.fp8).min, torch.finfo(self.fp8).max
        else: #INT8
            w_dtype, input_dtype, min_val, max_val = torch.int8, DType.INT8, torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max

        in_features, out_features = weight.shape[::-1]

        if(scales is None):
            #Quantize
            dtype = weight.dtype if(self.dtype is None) else self.dtype

            assert dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ], "Invalid weight dtype, should be floating point."

            qcgem_dtype = TORCH_TO_DTYPE[dtype]
            data_ptr = weight.data_ptr()
            weight = weight.to(dtype=torch.float32, copy=False, device=self.device)
            scales = weight.abs().amax(axis=1, keepdim=True) / max_val
            scales.clamp_(min=1e-6)
            W_q = (weight / scales).clamp_(min=min_val, max=max_val)
            if(w_dtype.is_floating_point):
                W_q = W_q.to(w_dtype)
            else:
                W_q = W_q.round_().to(w_dtype)
            if(data_ptr != weight.data_ptr()):
                del weight
                torch.cuda.empty_cache()
        else:
            #Pre-Quantized
            assert weight.dtype.itemsize == 1, "Invalid weight.dtype, should be 8-bit."
            if(self.dtype is None):
                dtype = scales.dtype if scales.dtype in [torch.float16, torch.bfloat16] else torch.float16
            else:
                dtype = self.dtype 
            W_q = weight.to(device=self.device)
            scales = scales.to(device=self.device) 
            qcgem_dtype = TORCH_TO_DTYPE[dtype]
        
        scales = scales.to(torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        qcgem_linear = QCGemLinearTriton(8, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype,
                        output_dtype=qcgem_dtype,
                        scaled_activations=True,
                        )

        qcgem_linear.pack(W_q, scales, zeros=None, bias=bias)

        #Post-scaling
        qcgem_linear.W_group_mode       = 0
        qcgem_linear.channel_scale_mode = 3
        return qcgem_linear

    def from_linear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(weight=linear_layer.weight, bias=linear_layer.bias)
        cleanup_linear(linear_layer, del_orig)
        return out_layer

class A8W8_int8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fp8 = False
A8W8_INT8_dynamic = A8W8_int8_dynamic

class A8W8_fp8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', dtype=None, fp8=default_fp8):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fp8 = fp8
A8W8_FP8_dynamic = A8W8_fp8_dynamic

#################################################################################################
#FP8 dynamic activations / W4 packed weights
#################################################################################################
class A8Wn_HQQ_INT_dynamic(A16Wn):
    def __init__(
        self,
        device="cuda:0",
        packing_bitwidth=None,
        dtype=None,
        post_scale=default_post_scale,
        fp8=default_fp8,
        fp32_scale=False,
        W_nbits=None,
    ):
        assert W_nbits is not None, "W_nbits argument should be eitehr 8 or 4, not None)."
        super().__init__()
        self.post_scale = post_scale
        self.device = device
        self.dtype = dtype
        self.packing_bitwidth = packing_bitwidth
        self.fp8 = fp8
        self.fp32_scale = fp32_scale
        self.W_nbits = W_nbits

    def from_weights(self, W_q, scales, zeros, bias=None):
        group_size = W_q.numel() // scales.numel()
        return self.from_weights_(
            W_q=W_q,
            scales=scales,
            zeros=zeros,
            W_nbits=self.W_nbits,
            group_size=group_size,
            bias=bias,
        )

    def from_weights_(self, W_q, scales, zeros, W_nbits, group_size, bias=None):
        if(isinstance(W_q, torch.nn.Parameter)):
            W_q = W_q.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if(self.dtype is None):
            dtype = scales.dtype if scales.dtype in [torch.float16, torch.bfloat16] else torch.float16
        else:
            dtype = self.dtype 

        assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid scales.dtype, should be floating point."
        qcgem_dtype = TORCH_TO_DTYPE[dtype]
        input_dtype = TORCH_TO_DTYPE[self.fp8]
            
        W_q = W_q.to(self.device)
        scales = scales.to(device=self.device, dtype=torch.float32 if self.fp32_scale else dtype) if (scales is not None) else None
        zeros = zeros.to(device=self.device, dtype=dtype) if (zeros is not None) else None
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        in_features, out_features = W_q.shape[::-1]

        qcgem_linear = QCGemLinearTriton(W_nbits, 
                        group_size=group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype, 
                        output_dtype=qcgem_dtype,
                        scaled_activations=True,
                        )

        qcgem_linear.pack(
            W_q,
            scales,
            zeros,
            bias=bias,
            packing_bitwidth=self.packing_bitwidth,
            fma_mode=False,
        )

        if(group_size == in_features):
            if(self.post_scale):
                qcgem_linear.W_group_mode = 1
                qcgem_linear.channel_scale_mode = 3
            else:
                qcgem_linear.W_group_mode = 3
                qcgem_linear.channel_scale_mode = 2

        return qcgem_linear

    def from_hqqlinear(self, hqq_layer, del_orig=True):
        assert hqq_layer.meta['axis'] == 1, 'Only axis==1 is supported.'

        self.device = hqq_layer.W_q.device

        W_nbits    = hqq_layer.meta['nbits']
        group_size = hqq_layer.meta["group_size"]
        if(group_size is None):
            group_size = hqq_layer.in_features

        W_q    = hqq_layer.unpack(dtype=torch.uint8).view(hqq_layer.meta['shape']) #Expects uint8 for Wn quantization!
        scales = hqq_layer.meta['scale'].clone()
        zeros  = hqq_layer.meta['zero'].clone()
        bias   = hqq_layer.bias.clone() if (hqq_layer.bias is not None) else None
        
        cleanup_linear(hqq_layer, del_orig)

        out_layer = self.from_weights_(
            W_q=W_q,
            scales=scales,
            zeros=zeros,
            W_nbits=W_nbits,
            group_size=group_size,
            bias=bias,
        )
        
        #Clean-up
        del W_q
        torch.cuda.empty_cache()

        return out_layer

class A8W4_HQQ_INT_dynamic(A8Wn_HQQ_INT_dynamic):
    def __init__(
        self,
        device="cuda:0",
        packing_bitwidth=None,
        dtype=None,
        post_scale=default_post_scale,
        fp8=default_fp8,
        fp32_scale=False,
    ):
        super().__init__(
            device=device,
            packing_bitwidth=packing_bitwidth,
            dtype=dtype,
            post_scale=post_scale,
            fp8=fp8,
            fp32_scale=fp32_scale,
            W_nbits=4,
        )


class A8W2_HQQ_INT_dynamic(A8Wn_HQQ_INT_dynamic):
    def __init__(
        self,
        device="cuda:0",
        packing_bitwidth=None,
        dtype=None,
        post_scale=default_post_scale,
        fp8=default_fp8,
        fp32_scale=False,
    ):
        super().__init__(
            device=device,
            packing_bitwidth=packing_bitwidth,
            dtype=dtype,
            post_scale=post_scale,
            fp8=fp8,
            fp32_scale=fp32_scale,
            W_nbits=2,
        )

#################################################################################################
#MXFP / NVFP dtypes with dynamic quant activations
#################################################################################################
class A8W8_MXFP_dynamic:
    def __init__(self, device='cuda:0', dtype=None, post_scale=True, fp8=default_fp8):
        self.device = device
        self.dtype = dtype
        self.mx_fp8_dtype = fp8
        self.quantizer_mx = None
        self.post_scale = post_scale
        self.W_nbits = 8

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]

        assert scales is not None, "Scales parameter cannot be None. Use from_linear() call to pre-quantize the weights."

        #Pre-Quantized
        assert weight.is_floating_point() and weight.itemsize == 1, f"Invalid weight.dtype, should be an MXPF8 dype, got {weight.dtype}."
        assert scales.dtype in [torch.float8_e8m0fnu, torch.uint8], f"Invalid scales.dtype, should be e8m0 / view(uint8), got {scales.dtype}."
        assert self.dtype is not None, f"Input dtype should be either torch.float16 or torch.bfloat16, not None."
        dtype = self.dtype 
        input_dtype = DType.MXFP8
        qcgem_dtype = TORCH_TO_DTYPE[dtype]
        group_size = 32

        W_q = weight.to(device=self.device)
        scales = scales.to(device=self.device) 
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        qcgem_linear = QCGemLinearTriton(self.W_nbits, 
                        group_size=group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype,
                        output_dtype=qcgem_dtype,
                        scaled_activations=True,
                        )

        qcgem_linear.pack(W_q, scales, zeros=None, bias=bias)

        #If post_scale==False, it will use mxfp8 microscales for the activations, otherwise channelwise post-scaling is used
        qcgem_linear.W_group_mode       = 0
        qcgem_linear.channel_scale_mode = 2 if self.post_scale else 4
        return qcgem_linear


    def from_linear(self, linear_layer, del_orig=True):
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        group_size = 32
        N, K = W.shape
        if(self.W_nbits == 8):
            W_q, scales = self.quantizer_mx.quantize_mxfp8(W, index=True, mx_fp8_dtype=self.mx_fp8_dtype)
        if(self.W_nbits == 4):
            W_q, scales = self.quantizer_mx.quantize_mxfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)
        
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights(weight=W_q, scales=scales, bias=bias)

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

class A8Wn_MXFP_dynamic:
    def __init__(self, device='cuda:0', dtype=None, post_scale=True, fp8=default_fp8, W_nbits=None):
        assert W_nbits is not None, "W_nbits argument should be eitehr 8 or 4, not None)."
        self.device = device
        self.dtype = dtype
        self.mx_fp8_dtype = fp8
        self.quantizer_mx = None
        self.post_scale = post_scale
        self.W_nbits = W_nbits

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]

        assert scales is not None, "Scales parameter cannot be None. Use from_linear() call to pre-quantize the weights."

        #Pre-Quantized
        if(self.W_nbits == 8):
            assert weight.is_floating_point() and weight.itemsize == 1, f"Invalid weight.dtype, should be an MXPF8 valid dtype, got {weight.dtype}."
        if(self.W_nbits == 4):
            assert weight.dtype in [torch.uint8], f"Invalid weight.dtype, should be an MXPF8 (FP8 dtype) valid dtype, got {weight.dtype}."
        assert scales.dtype in [torch.float8_e8m0fnu, torch.uint8], f"Invalid scales.dtype, should be e8m0 / view(uint8), got {scales.dtype}."
        assert self.dtype is not None, f"Input dtype should be either torch.float16 or torch.bfloat16, not None."
        dtype = self.dtype 
        input_dtype = DType.MXFP8
        qcgem_dtype = TORCH_TO_DTYPE[dtype]
        group_size = 32

        W_q = weight.to(device=self.device)
        scales = scales.to(device=self.device) 
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        qcgem_linear = QCGemLinearTriton(self.W_nbits, 
                        group_size=group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype,
                        output_dtype=qcgem_dtype,
                        scaled_activations=True,
                        )

        qcgem_linear.pack(W_q, scales, zeros=None, bias=bias)

        #If post_scale==False, it will use mxfp8 microscales for the activations, otherwise channelwise post-scaling is used
        qcgem_linear.W_group_mode       = 0
        qcgem_linear.channel_scale_mode = 2 if self.post_scale else 4
        return qcgem_linear


    def from_linear(self, linear_layer, del_orig=True):
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        group_size = 32
        N, K = W.shape
        if(self.W_nbits == 8):
            W_q, scales = self.quantizer_mx.quantize_mxfp8(W, index=True, mx_fp8_dtype=self.mx_fp8_dtype)
        if(self.W_nbits == 4):
            W_q, scales = self.quantizer_mx.quantize_mxfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)
        
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights(weight=W_q, scales=scales, bias=bias)

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

class A8W8_MXFP_dynamic(A8Wn_MXFP_dynamic):
    def __init__(self, device='cuda:0', dtype=None, post_scale=True, fp8=default_fp8):
        super().__init__(device=device, dtype=dtype, post_scale=post_scale, fp8=fp8, W_nbits=8)

class A8W4_MXFP_dynamic(A8Wn_MXFP_dynamic):
    def __init__(self, device='cuda:0', dtype=None, post_scale=True, fp8=default_fp8):
        super().__init__(device=device, dtype=dtype, post_scale=post_scale, fp8=fp8, W_nbits=4)

class A4W4_MXFP_dynamic:
    def __init__(self, device='cuda:0', dtype=None):
        self.device = device
        self.dtype = dtype
        self.quantizer_mx = None
        self.W_nbits = 4
        self.group_size = 32
        self.input_dtype = DType.MXFP4

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]

        assert scales is not None, "Scales parameter cannot be None. Use from_linear() call to pre-quantize the weights."

        #Pre-Quantized
        assert weight.dtype in [torch.uint8], f"Invalid weight.dtype, should be an MXPF8 valid dtype, got {weight.dtype}."
        assert scales.dtype in [torch.float8_e8m0fnu, torch.uint8], f"Invalid scales.dtype, should be e8m0 / view(uint8), got {scales.dtype}."
        assert self.dtype is not None, f"Input dtype should be either torch.float16 or torch.bfloat16, not None."
        assert self.group_size == 32, f"Only group_size=16 is supported for MXFP4, got {self.group_size}"

        dtype = self.dtype 
        qcgem_dtype = TORCH_TO_DTYPE[dtype]

        W_q = weight.to(device=self.device)
        scales = scales.to(device=self.device) 
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        qcgem_linear = QCGemLinearTriton(self.W_nbits, 
                        group_size=self.group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=self.input_dtype,
                        output_dtype=qcgem_dtype,
                        scaled_activations=True,
                        )

        qcgem_linear.pack(W_q, scales, zeros=None, bias=bias)
        qcgem_linear.W_group_mode       = 0
        qcgem_linear.channel_scale_mode = 4
        return qcgem_linear


    def from_linear(self, linear_layer, del_orig=True):
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        N, K = W.shape
        W_q, scales = self.quantizer_mx.quantize_mxfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // self.group_size)
        
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights(weight=W_q, scales=scales, bias=bias)

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

class A4W4_NVFP_dynamic:
    def __init__(self, device='cuda:0', dtype=None):
        self.device = device
        self.dtype = dtype
        self.quantizer_mx = None
        self.W_nbits = 4
        self.group_size = 16
        self.input_dtype = DType.NVFP4

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]

        assert scales is not None, "Scales parameter cannot be None. Use from_linear() call to pre-quantize the weights."

        #Pre-Quantized
        assert weight.dtype in [torch.uint8], f"Invalid weight.dtype, should be an MXPF8 dtype, got {weight.dtype}."
        assert scales.dtype in [torch.float8_e4m3fn], f"Invalid scales.dtype, should be float8_e4m3fn, got {scales.dtype}."
        assert self.dtype is not None, f"Input dtype should be either torch.float16 or torch.bfloat16, not None."
        assert self.group_size == 16, f"Only group_size=16 is supported for NVFP4, got {self.group_size}"
         
        dtype = self.dtype
        qcgem_dtype = TORCH_TO_DTYPE[dtype]
        
        W_q = weight.to(device=self.device)
        scales = scales.to(device=self.device) 
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        qcgem_linear = QCGemLinearTriton(self.W_nbits, 
                        group_size=self.group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=self.input_dtype,
                        output_dtype=qcgem_dtype,
                        scaled_activations=True,
                        )

        qcgem_linear.pack(W_q, scales, zeros=None, bias=bias)
        qcgem_linear.W_group_mode       = 0
        qcgem_linear.channel_scale_mode = 4
        return qcgem_linear


    def from_linear(self, linear_layer, del_orig=True):
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        N, K = W.shape
        W_q, scales = self.quantizer_mx.quantize_nvfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // self.group_size)
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights(weight=W_q, scales=scales, bias=bias)

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

#################################################################################################
#BitNet
#################################################################################################
class A16W158_INT:
    def __init__(self, device='cuda:0', dtype=None, fp32_scale=True):
        self.device = device
        self.dtype = dtype
        self.fp32_scale = fp32_scale

    def from_weights(self, weight, weight_scale, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        dtype = weight.dtype if(self.dtype is None) else self.dtype

        assert dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "Invalid weight.dtype, should be floating point."

        weight        = weight.to(dtype=dtype, device=self.device)
        W_q           = (weight + 1).to(torch.uint8)
        bias          = bias if bias is not None else None
        dtype         = weight.dtype
        qcgem_dtype = TORCH_TO_DTYPE[dtype]
        out_features  = W_q.shape[0]
        in_features   = W_q.shape[1]
        scales        = torch.ones((out_features, 1), dtype=torch.float32) * weight_scale.item() 

        qcgem_linear = QCGemLinearTriton(2, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=qcgem_dtype, 
                        output_dtype=qcgem_dtype, 
                        scaled_activations=False,
                        )

        scales = scales.to(dtype=torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        qcgem_linear.pack(W_q, scales=scales, zeros=1, bias=bias)
        #post-scale
        qcgem_linear.W_group_mode       = 1 #shift only
        qcgem_linear.channel_scale_mode = 1 #weight-only
        return qcgem_linear

    def from_bitlinear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(
            weight=linear_layer.weight,
            weight_scale=linear_layer.weight_scale,
            bias=linear_layer.bias,
        )
        cleanup_linear(linear_layer, del_orig)
        return out_layer

class A8W158_INT_dynamic:
    def __init__(self, device="cuda:0", dtype=None, fp32_scale=True):
        self.device = device
        self.dtype = dtype
        self.fp32_scale = fp32_scale

    def from_weights(self, weight, weight_scale, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        dtype = weight.dtype if(self.dtype is None) else self.dtype

        assert dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "Invalid weight.dtype, should be floating point."

        weight        = weight.to(device=self.device, dtype=dtype)
        W_q           = (weight + 1).to(torch.uint8)
        dtype         = weight.dtype
        qcgem_dtype = TORCH_TO_DTYPE[dtype]
        out_features  = W_q.shape[0]
        in_features   = W_q.shape[1]
        scales        = torch.ones((out_features, 1), dtype=torch.float32, device=self.device) * weight_scale.item()
        input_dtype   = DType.INT8

        qcgem_linear = QCGemLinearTriton(2, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype, 
                        output_dtype=qcgem_dtype, 
                        scaled_activations=True,
                        )

        scales = scales.to(dtype=torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        qcgem_linear.pack(W_q, scales=scales, zeros=1, bias=bias)

        #post-scale 
        qcgem_linear.W_group_mode       = 1 #shift only
        qcgem_linear.channel_scale_mode = 3 #activations + weight

        return qcgem_linear

    def from_bitlinear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(
            weight=linear_layer.weight,
            weight_scale=linear_layer.weight_scale,
            bias=linear_layer.bias,
        )
        cleanup_linear(linear_layer, del_orig)
        return out_layer

#################################################################################################
#Warm-up function
#################################################################################################
default_batch_sizes = sorted(list(set(M_MAPPING)))[::-1]
def warmup(
    processor,
    shapes: list[tuple],
    batch_sizes: list[int] = default_batch_sizes,
    group_size: int = 64,
    dtype: torch.dtype = torch.float16,
):
    """
    processor = A8W8_INT8_dynamic()
    warmup(processor, shapes=[(4096, 4096), (4096, 2048)], batch_sizes=[1, 4, 8, ..])
    `group_size` is only used for hqq quantization.
    """

    if(hasattr(processor, "from_hqqlinear")):
        try:
            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        except ImportException:
            print("This processor requires the `hqq` package. Install it via `pip install hqq`.")
    else:
        class _NoHQQ: pass
        HQQLinear = _NoHQQ


    device = torch.device(torch.cuda.current_device())
    
    for shape in shapes:
        out_features, in_features = shape

        linear = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            dtype=dtype,
            device=device,
        )

        if(hasattr(processor, 'from_hqqlinear')):
            W_nbits        = processor.W_nbits
            quant_config   = BaseQuantizeConfig(nbits=W_nbits, group_size=group_size, axis=1)
            linear         = HQQLinear(linear, quant_config=quant_config, compute_dtype=dtype, device=device)
            qcgem_linear = processor.from_hqqlinear(linear)
        else:
            qcgem_linear = processor.from_linear(linear)

        for batch_size in tqdm(batch_sizes):
            _ = qcgem_linear(torch.randn((batch_size, in_features), dtype=dtype, device=device) / 100.)
            torch.cuda.synchronize()

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

