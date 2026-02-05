import os
import sys
import ast
from typing import List, Optional
import re
import json  
from openai import OpenAI

def mock_str_gen(file_content: str) -> str:
    """
    根据文件内容生成注释字符串
    这是一个示例实现，你可以根据需要修改这个函数
    """
    client = OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url="https://api.siliconflow.cn/v1"
    )
    
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[
            {
                "role": "system", 
                "content": """
You are software engineer.
You will generate comment for code block ref pytorch style.
You only need output comments part.
Your response including description, Args, Keyword arguments and Examples.

Here is an example pytorch comment:            
add(input, other, *, alpha=1, out=None) -> Tensor

Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

.. math::
    \\text{{out}}_i = \\text{{input}}_i + \\text{{alpha}} \\times \\text{{other}}_i

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    {input}
    other (Tensor or Number): the tensor or number to add to :attr:`input`.

Keyword arguments:
    alpha (Number): the multiplier for :attr:`other`.
    {out}

Examples::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
    >>> torch.add(a, 20)
    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

    >>> b = torch.randn(4)
    >>> b
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
    >>> c = torch.randn(4, 1)
    >>> c
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
    >>> torch.add(b, c, alpha=10)
    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
            [-18.6971, -18.0736, -17.0994, -17.3216],
            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
                """
            },
            {
                "role": "user", 
                "content": "Code block: \n" + file_content
            }
        ]
    )
    return response.choices[0].message.content
    
def insert_comment_to_file(file_path: str, comment: str) -> None:
    """
    将注释插入到Python文件的头部
    
    Args:
        file_path: 文件路径
        comment: 要插入的注释内容
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 将注释转换为Python注释格式
    comment_lines = comment.split('\n')
    comment_block = '# ' + '\n# '.join(comment_lines)
    
    # 检查文件是否有shebang（#!/usr/bin/env python等）
    if content.startswith('#!'):
        # 找到第一行结束的位置
        first_newline = content.find('\n')
        if first_newline != -1:
            # 在shebang后插入注释
            new_content = content[:first_newline + 1] + '\n' + comment_block + '\n\n' + content[first_newline + 1:]
        else:
            # 如果文件只有shebang一行
            new_content = content + '\n\n' + comment_block + '\n'
    else:
        # 没有shebang，直接在文件开头插入
        new_content = comment_block + '\n\n' + content
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"已更新文件：{file_path}")


def process_python_files(directory: str, max_files: Optional[int] = None) -> None:
    """
    处理目录下的Python文件
    
    Args:
        directory: 要处理的目录路径
        max_files: 最大处理文件数量，None表示处理所有文件
    """
    # 获取目录下的所有Python文件
    python_files = []
    
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.py') and filename != '__init__.py':
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    python_files.append((filename, filepath))
    except FileNotFoundError:
        print(f"错误：目录 '{directory}' 不存在")
        return
    except PermissionError:
        print(f"错误：没有权限访问目录 '{directory}'")
        return
    
    # 按字母顺序排序
    python_files.sort(key=lambda x: x[0].lower())
    
    # 限制文件数量
    if max_files is not None and max_files > 0:
        python_files = python_files[:max_files]
    
    if not python_files:
        print(f"在目录 '{directory}' 中未找到Python文件（已跳过__init__.py）")
        return
    
    print(f"找到 {len(python_files)} 个Python文件需要处理：")
    for i, (filename, _) in enumerate(python_files, 1):
        print(f"  {i}. {filename}")
    
    # 处理每个文件
    processed_count = 0
    for filename, filepath in python_files:
        try:
            # 读取文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # 生成注释
            comment = mock_str_gen(file_content)
            
            # 插入注释到文件头部
            insert_comment_to_file(filepath, comment)
            
            processed_count += 1
            print(f"进度：{processed_count}/{len(python_files)}")
            
        except UnicodeDecodeError:
            print(f"警告：无法以UTF-8编码读取文件 {filename}，跳过")
        except Exception as e:
            print(f"错误：处理文件 {filename} 时出错：{e}")
    
    print(f"\n处理完成！共处理了 {processed_count} 个文件")


def main():
    """
    主函数：处理命令行参数并执行
    """
    if len(sys.argv) < 2:
        print("用法: python script.py <目录路径> [最大文件数量]")
        print("示例: python script.py ./src 10")
        print("示例: python script.py ./src (处理所有文件)")
        return
    
    directory = sys.argv[1]
    
    # 处理最大文件数量参数
    max_files = None
    if len(sys.argv) >= 3:
        try:
            max_files = int(sys.argv[2])
            if max_files <= 0:
                print("警告：最大文件数量必须为正整数，将处理所有文件")
                max_files = None
        except ValueError:
            print("警告：最大文件数量必须是整数，将处理所有文件")
            max_files = None
    
    # 处理文件
    process_python_files(directory, max_files)


if __name__ == "__main__":
    main()