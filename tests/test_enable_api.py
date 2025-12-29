import flag_gems
import pytest
import torch
import os
import re
import atexit


def get_path_log_name():
   return "./gems_enable_test.log"

def save_log_file(ori_log, sv_log):
   if os.path.exists(ori_log):
      with open(ori_log, 'rb') as f_in:
         content = f_in.read()

      content = content.lstrip(b'\x00')

      with open(sv_log, 'wb') as f_out:
         f_out.write(content)

      with open(ori_log, 'wb') as f:
         pass

def cleanup_log_file():
    path_file = get_path_log_name()
    if os.path.exists(path_file):
        os.remove(path_file)


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_log():
    atexit.register(cleanup_log_file)
    yield


@pytest.mark.enable
def test_enable():
   path_file = get_path_log_name()
   with flag_gems.use_gems(record=True, path=path_file):
      a = torch.tensor([1.0, 2.0, 3.0], device=flag_gems.device)
      b = torch.tensor([4.0, 5.0, 6.0], device=flag_gems.device)
      _ = a + b
      _ = a * b
      _ = flag_gems.sum(a)
      _ = torch.sum(a)

   log_file = f"./gems_enable_all_ops.log"
   save_log_file(path_file, log_file)

   assert os.path.exists(log_file), f"Log file {log_file} not found"
   with open(log_file, 'r') as f:
      log_content = f.read()

   pattern = r"flag_gems\.ops\.(\w+):"
   found_ops = set(re.findall(pattern, log_content))
   expected_ops = ["add", "mul", "sum"]
   for op in expected_ops:
      assert op in found_ops, f"Expected op '{op}' not found in log file"


@pytest.mark.enable_with_exclude
@pytest.mark.parametrize("exclude_op", [["mul"], ["mul", "add"]])
def test_enable_with_exclude(exclude_op):
   path_file = get_path_log_name()
   with flag_gems.use_gems(exclude=exclude_op, record=True, path=path_file):
      a = torch.tensor([1.0, 2.0, 3.0], device=flag_gems.device)
      b = torch.tensor([4.0, 5.0, 6.0], device=flag_gems.device)
      _ = a + b
      _ = a * b
      _ = flag_gems.sum(a)
      _ = torch.sum(a)
  
   op_names_str = "_".join(exclude_op).replace(".", "_").replace("-", "_")
   log_file = f"./gems_enable_without_{op_names_str}.log"
   save_log_file(path_file, log_file)

   assert os.path.exists(log_file), f"Log file {log_file} not found"
   with open(log_file, 'r') as f:
      log_content = f.read()

   pattern = r"flag_gems\.ops\.(\w+):"
   found_ops = set(re.findall(pattern, log_content))
   for op in found_ops:
      assert op not in exclude_op, f"Found excluded op '{op}' in log file."


@pytest.mark.only_enable
@pytest.mark.parametrize("include_op", [["sum"], ["mul", "sum"], ["sum", "mul", "add"]])
def test_only_enable(include_op):
   path_file = get_path_log_name()
   with flag_gems.use_gems(include=include_op, record=True, path=path_file):
      a = torch.tensor([1.0, 2.0, 3.0], device=flag_gems.device)
      b = torch.tensor([4.0, 5.0, 6.0], device=flag_gems.device)
      _ = a + b
      _ = a * b
      _ = flag_gems.sum(a)
      _ = torch.sum(a)

   op_names_str = "_".join(include_op).replace(".", "_").replace("-", "_")
   log_file = f"./gems_only_enable_{op_names_str}.log"
   save_log_file(path_file, log_file)

   assert os.path.exists(log_file), f"Log file {log_file} not found"
   with open(log_file, 'r') as f:
      log_content = f.read()

   pattern = r"flag_gems\.ops\.(\w+):"
   found_ops = set(re.findall(pattern, log_content))
   for op in found_ops:
      assert op in include_op, f"Found unexpected op '{op}' in log file. Allowed op: {include_op}"
