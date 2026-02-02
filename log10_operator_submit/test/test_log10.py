import torch
import sys
sys.path.append(".")  # 添加上级目录到路径
from operator.log10_ops import log10, log10_, log10_out

def test_precision():
    """精度验证：与 PyTorch 原生 log10 对比，误差 < 1e-6"""
    # 测试多维度、多数值场景
    test_cases = [
        torch.tensor([1.0, 10.0, 100.0, 0.1, 2.0], dtype=torch.float32),
        torch.tensor([1000.0, 0.01, 5.0], dtype=torch.float64),
        torch.randn((5, 5), dtype=torch.float32).abs() + 0.1,  # 随机正数张量
        torch.randn((2, 3, 4), dtype=torch.float64).abs() + 0.1
    ]
    
    for idx, x in enumerate(test_cases):
        # 普通版验证
        my_log10 = log10(x)
        torch_log10 = torch.log10(x)
        assert torch.allclose(my_log10, torch_log10, atol=1e-6), \
            f"普通版测试用例 {idx} 精度不达标"
        
        # In-place 版验证
        x_inplace = x.clone()
        x_torch = x.clone()
        log10_(x_inplace)
        x_torch.log10_()
        assert torch.allclose(x_inplace, x_torch, atol=1e-6), \
            f"In-place 版测试用例 {idx} 精度不达标"
        
        # Out 版验证
        out_my = torch.empty_like(x)
        out_torch = torch.empty_like(x)
        log10_out(x, out=out_my)
        torch.log10(x, out=out_torch)
        assert torch.allclose(out_my, out_torch, atol=1e-6), \
            f"Out 版测试用例 {idx} 精度不达标"
    
    print("✅ 所有精度测试通过（误差 < 1e-6）")

def test_performance():
    """性能验证：测试大张量下的执行效率"""
    # 模拟竞赛性能测试场景（1000x1000 浮点张量）
    x = torch.randn((1000, 1000), dtype=torch.float32).abs() + 0.1
    
    # 普通版耗时
    torch.cuda.synchronize() if x.is_cuda else None
    if x.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        log10(x)
        torch.cuda.synchronize()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        print(f"普通版耗时：{start.elapsed_time(end):.4f} ms")
    else:
        from timeit import timeit
        # CPU 版耗时（运行1000次取平均）
        avg_time = timeit(lambda: log10(x), number=1000) / 1000 * 1000  # 转 ms
        print(f"普通版平均耗时：{avg_time:.4f} ms")
    
    print("✅ 性能测试完成")

def test_exception():
    """异常处理验证：确保非法输入触发正确异常"""
    # 测试1：负数/零输入触发 ValueError
    x = torch.tensor([-1.0, 0.0])
    try:
        log10(x)
        # 若未抛异常，触发断言失败
        assert False, "未触发负数/零输入异常"
    except ValueError as e:
        # 确认异常信息包含关键提示
        assert "严格正数张量" in str(e), "异常信息不符合预期"
    
    # 测试2：out 张量 dtype 不匹配触发 ValueError
    out = torch.empty((2,), dtype=torch.float64)
    x_float32 = torch.tensor([1.0, 10.0], dtype=torch.float32)
    try:
        log10_out(x_float32, out=out)
        assert False, "未触发 dtype 不匹配异常"
    except ValueError:
        pass
    
    # 测试3：in-place 版负数输入触发异常
    x_inplace = torch.tensor([-5.0])
    try:
        log10_(x_inplace)
        assert False, "in-place 版未触发负数输入异常"
    except ValueError:
        pass
    
    print("✅ 异常处理测试通过")

if __name__ == "__main__":
    # 优先测试 CPU，可选测试 CUDA（如有 GPU）
    test_precision()
    test_exception()
    test_performance()
    
    # 可选：CUDA 测试（注释掉也不影响提交）
    if torch.cuda.is_available():
        print("\n=== CUDA 版本测试 ===")
        x_cuda = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float32).cuda()
        my_log10_cuda = log10(x_cuda)
        torch_log10_cuda = torch.log10(x_cuda)
        assert torch.allclose(my_log10_cuda, torch_log10_cuda, atol=1e-6)
        print("✅ CUDA 版本测试通过")
    
    print("\n🎉 所有测试全部通过，可提交竞赛！")