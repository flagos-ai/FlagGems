#!/usr/bin/env python3
"""
批量修复类别1的测试文件（180个）
模式: ref_xxx = xxx.clone() → ref_xxx = to_reference(xxx)
"""

import os
import re
from pathlib import Path

# to_reference 函数定义
TO_REFERENCE_FUNC = '''
def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp
'''

# 类别1文件列表（180个，从统计中得出）
TYPE1_FILES = [
    "_adaptive_avg_pool2d_test.py",
    "_adaptive_avg_pool3d_test.py",
    "_reshape_alias_test.py",
    "_safe_softmax_test.py",
    "_unsafe_view_test.py",
    "_upsample_nearest_exact1d_test.py",
    "_upsample_nearest_exact3d_test.py",
    "abs_test.py",
    "absolute_test.py",
    "addcdiv_test.py",
    "addcmul__test.py",
    "alias_copy_test.py",
    "amin_test.py",
    "arccosh__test.py",
    "arccosh_test.py",
    "arcsinh__test.py",
    "arcsinh_test.py",
    "arctanh__test.py",
    "arctanh_test.py",
    "as_strided__test.py",
    "as_strided_copy_test.py",
    "as_strided_scatter_test.py",
    "as_strided_test.py",
    "asinh__test.py",
    "atanh__test.py",
    "ceil__test.py",
    "celu_test.py",
    "clamp_max__test.py",
    "clamp_min__test.py",
    "copy__test.py",
    "cos__test.py",
    "cosh__test.py",
    "deg2rad__test.py",
    "deg2rad_test.py",
    "diag_test.py",
    "digamma__test.py",
    "elu_test.py",
    "eq__test.py",
    "erf__test.py",
    "erfinv__test.py",
    "erfinv_test.py",
    "exp2__test.py",
    "exp2_test.py",
    "exp__test.py",
    "expand_test.py",
    "eye_test.py",
    "fft_fftshift_test.py",
    "fft_ifftshift_test.py",
    "fix__test.py",
    "fix_test.py",
    "floor__test.py",
    "fmin_test.py",
    "frac__test.py",
    "frac_test.py",
    "ge__test.py",
    "gelu__test.py",
    "glu_test.py",
    "greater__test.py",
    "greater_equal__test.py",
    "hardshrink_test.py",
    "hardsigmoid__test.py",
    "hardsigmoid_test.py",
    "hardswish__test.py",
    "hardswish_test.py",
    "hardtanh__test.py",
    "hardtanh_test.py",
    "heaviside__test.py",
    "heaviside_test.py",
    "hinge_embedding_loss_test.py",
    "huber_loss_test.py",
    "hypot__test.py",
    "hypot_test.py",
    "i0__test.py",
    "i0_test.py",
    "im2col_test.py",
    "le__test.py",
    "leaky_relu__test.py",
    "leaky_relu_test.py",
    "lerp__test.py",
    "less__test.py",
    "less_equal__test.py",
    "lgamma__test.py",
    "lift_fresh_copy_test.py",
    "lift_fresh_test.py",
    "lift_test.py",
    "log10__test.py",
    "log1p__test.py",
    "log2_test.py",
    "log__test.py",
    "logaddexp2_test.py",
    "logaddexp_test.py",
    "logical_not__test.py",
    "logical_xor__test.py",
    "logit__test.py",
    "logit_test.py",
    "lt__test.py",
    "margin_ranking_loss_test.py",
    "masked_fill_test.py",
    "masked_scatter_test.py",
    "masked_select_test.py",
    "maximum_test.py",
    "mse_loss_test.py",
    "multiply_test.py",
    "mv_test.py",
    "native_dropout_backward_test.py",
    "ne__test.py",
    "neg__test.py",
    "negative__test.py",
    "negative_test.py",
    "new_ones_test.py",
    "not_equal__test.py",
    "permute_copy_test.py",
    "permute_test.py",
    "pixel_shuffle_test.py",
    "pixel_unshuffle_test.py",
    "positive_test.py",
    "prelu_test.py",
    "rad2deg__test.py",
    "reciprocal__test.py",
    "reciprocal_test.py",
    "reflection_pad1d_test.py",
    "reflection_pad2d_test.py",
    "reflection_pad3d_test.py",
    "relu6_test.py",
    "relu__test.py",
    "replication_pad1d_test.py",
    "replication_pad2d_test.py",
    "replication_pad3d_test.py",
    "reshape_test.py",
    "rrelu_with_noise_backward_test.py",
    "rsqrt__test.py",
    "selu__test.py",
    "selu_test.py",
    "sgn__test.py",
    "sgn_test.py",
    "sigmoid__test.py",
    "sigmoid_test.py",
    "sign__test.py",
    "sign_test.py",
    "silu__test.py",
    "silu_test.py",
    "sin__test.py",
    "sinc__test.py",
    "sinc_test.py",
    "sinh__test.py",
    "smooth_l1_loss_test.py",
    "soft_margin_loss_test.py",
    "softplus_test.py",
    "softshrink_test.py",
    "special_i0e_test.py",
    "special_i1_test.py",
    "special_xlog1py_test.py",
    "square__test.py",
    "square_test.py",
    "squeeze_copy_test.py",
    "t__test.py",
    "t_copy_test.py",
    "t_test.py",
    "take_test.py",
    "threshold__test.py",
    "threshold_test.py",
    "trace_test.py",
    "transpose_copy_test.py",
    "tril__test.py",
    "tril_test.py",
    "triu__test.py",
    "triu_test.py",
    "trunc_test.py",
    "unsqueeze__test.py",
    "unsqueeze_copy_test.py",
    "unsqueeze_test.py",
    "upsample_nearest1d_test.py",
    "upsample_nearest3d_test.py",
    "view_as_real_test.py",
    "xlogy__test.py",
    "xlogy_test.py",
    "zero__test.py",
    "zero_test.py",
    "zeros_like_test.py",
]

def fix_test_file(filepath):
    """修复单个测试文件"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. 修改 import 部分，添加 TO_CPU
    # 查找 from tests.accuracy_utils import gems_assert_close
    if 'from tests.accuracy_utils import gems_assert_close' in content:
        # 检查是否已经有 TO_CPU
        if 'TO_CPU' not in content.split('def to_reference')[0]:  # 只检查函数定义之前的部分
            content = re.sub(
                r'from tests\.accuracy_utils import gems_assert_close(?!\s*,\s*TO_CPU)',
                'from tests.accuracy_utils import gems_assert_close, TO_CPU',
                content
            )
    
    # 2. 在 ImportError 的 except 块中添加 TO_CPU fallback
    # 查找 except ImportError: 后面紧跟的注释行
    if 'except ImportError:' in content and 'TO_CPU = False' not in content:
        content = re.sub(
            r'(except ImportError:\s*\n\s*#[^\n]*\n)',
            r'\1    TO_CPU = False  # fallback\n',
            content
        )
    
    # 3. 在导入之后、第一个 @pytest 之前添加 to_reference 函数
    # 找到第一个 @pytest.mark 的位置
    if 'def to_reference(' not in content:
        pytest_match = re.search(r'\n\n@pytest\.mark\.', content)
        if pytest_match:
            insert_pos = pytest_match.start() + 1  # 保留一个空行
            content = content[:insert_pos] + TO_REFERENCE_FUNC + '\n' + content[insert_pos:]
    
    # 4. 替换 ref_xxx = xxx.clone() 为 ref_xxx = to_reference(xxx)
    # 但是：只在正确性测试函数中替换，不在性能测试函数中替换
    
    # 策略：逐行处理，跟踪当前所在的函数
    lines = content.split('\n')
    new_lines = []
    in_performance_test = False
    
    for line in lines:
        # 检测函数定义
        if line.strip().startswith('def test_'):
            # 判断是否是性能测试函数
            if 'benchmark' in line or 'performance' in line:
                in_performance_test = True
            else:
                in_performance_test = False
        
        # 只在非性能测试函数中替换 ref_xxx = xxx.clone()
        if not in_performance_test:
            line = re.sub(
                r'^(\s+)(ref_\w+)\s*=\s*(\w+)\.clone\(\)(?!\s*\.)$',
                r'\1\2 = to_reference(\3)',
                line
            )
        
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    # 5. 如果内容有变化，写回文件
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    else:
        return False

def main():
    """主函数：批量处理类别1的测试文件"""
    import sys
    
    test_dir = Path(__file__).parent / 'tests' / 'experimental_ops'
    
    if not test_dir.exists():
        print(f"Error: Directory not found: {test_dir}")
        return
    
    # 测试模式：只处理前3个文件
    test_mode = '--test' in sys.argv
    
    if test_mode:
        files_to_process = TYPE1_FILES[:3]
        print(f"🧪 测试模式：只处理前3个文件")
    else:
        files_to_process = TYPE1_FILES
        print(f"🚀 生产模式：处理所有 {len(TYPE1_FILES)} 个文件")
    
    print(f"\n{'='*70}")
    
    fixed_count = 0
    skipped_count = 0
    
    for filename in files_to_process:
        filepath = test_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  跳过（不存在）: {filename}")
            skipped_count += 1
            continue
        
        try:
            if fix_test_file(filepath):
                print(f"✅ 已修改: {filename}")
                fixed_count += 1
            else:
                print(f"⏭️  跳过（无需修改）: {filename}")
                skipped_count += 1
        except Exception as e:
            print(f"❌ 错误: {filename} - {e}")
            skipped_count += 1
    
    print(f"\n{'='*70}")
    print(f"📊 统计:")
    print(f"   目标文件: {len(files_to_process)}")
    print(f"   已修改: {fixed_count}")
    print(f"   跳过: {skipped_count}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
