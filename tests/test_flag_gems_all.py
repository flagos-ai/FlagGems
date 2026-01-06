import os, sys
import time
import argparse
import subprocess
import json

##test_set_name必须在json文件中有定义
def read_json_ops_and_tasks(file_path, test_set_name=None):
    all_op_list = []
    test_op_list = None
    all_task_dict = {}
    print(f"[Triton CI Run Flaggems OPs][info]{file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            all_op_list = json_data.get('all_ops', [])
            if test_set_name is not None:
                test_op_list = json_data.get(test_set_name, [])
            all_task_dict = json_data.get('all_tasks', {})

    except FileNotFoundError:
        print(f"[Triton CI Run Flaggems OPs][error] {file_path} not found!")
    except json.JSONDecodeError:
        print(f"[Triton CI Run Flaggems OPs][error] {file_path} data decode error!")
    except Exception as e:
        print(f"[Triton CI Run Flaggems OPs][error] {file_path} read json fail!")

    return all_op_list, test_op_list, all_task_dict


def run_task(task, args, log_dir = "dump/flag_gems"):
    test_file = task.split("::")[0]
    test_func = task.split("::")[1]
    log_file = os.path.join(log_dir, f"{test_file}-{test_func}.log")
    env = os.environ.copy()
    env["TXDA_VISIBLE_DEVICES"] = args.device
    env["TRITON_ALWAYS_COMPILE"] = "1"
    env["PRECISION_PRIORITY"] = "1"

    file_dir = os.path.dirname(os.path.abspath(__file__))
    triton_dir = file_dir+"/../../triton/third_party/tsingmicro/scripts/"
    if os.path.exists(triton_dir) is False:
        triton_dir = file_dir+"/../scripts/"
    if os.path.exists(triton_dir) is False:
        print(f"[Triton CI Run Flaggems OPs][error] {triton_dir} not exists!")
        return -1
    cmd = [f"{triton_dir}/run_tsingmicro.sh", "pytest", f"{file_dir}/{task}", "--ref", "cpu"]
    if args.quick:
        cmd.append("--mode")
        cmd.append("quick")
    print(f"[Triton CI Run Flaggems OPs][info]cmd:{' '.join(cmd)}")

    with open(log_file, "wb") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FlagGems OP accuracy test for Triton daily CI.")
    parser.add_argument("--test_set",
                        type = str,
                        help = "op set name to test, define in flag_gems_ci_ops.json, e.g. --test_set all_ops",
                        default = None)
    parser.add_argument("--start",
                        type = int,
                        help = "start index_id to test in test_set",
                        default= 0)
    parser.add_argument("--end",
                        type = int,
                        help = "end index_id to test in test_set",
                        default = sys.maxsize)
    parser.add_argument("--op",
                        type = str,
                        help = "op name to test, e.g. --op abs",
                        default = None)
    parser.add_argument("--func",
                        type = str,
                        help = "function to test,e.g. --func test_unary_pointwise_ops.py::test_accuracy_abs",
                        default = None)
    parser.add_argument("--device",
                        type = str,
                        help = "set device id for use, when test on multi cards machine",
                        default= "0")
    parser.add_argument('--quick', action='store_true', default=False,
                        help='run tests on quick mode')
    args = parser.parse_args()
    
    status = 0
    for param in [args.test_set, args.op, args.func]:
        if param is not None:
            status += 1
    if status == 0:
        print("error: --test_set, --op, or --func must be set one!")
        sys.exit(-1)
    elif status >= 2:
        print("error: --test_set, --op, or --func only support set one!")
        sys.exit(-1)
    
    log_dir = "dump/flag_gems"
    os.makedirs(log_dir, exist_ok=True)
    if args.test_set is not None or args.op is not None:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = file_dir + "/flag_gems_ci_ops.json"
        all_op_list, test_op_list, all_tasks = read_json_ops_and_tasks(json_file_path, args.test_set)
        if test_op_list is None and args.op is not None:
            test_op_list = [args.op]
        
        args.end = min(args.end, len(test_op_list)-1)
        failed_task_count = 0
        succ_task_count = 0
        failed_op_count = 0
        succ_op_count = 0
        for idx, op in enumerate(test_op_list):
            if idx < args.start:
                continue
            if idx > args.end:
                break
            if op in all_tasks:
                print(f"[Triton CI Run Flaggems OPs][info] {idx}: {op} begin to test...")
                failed_op_func_count = 0
                succ_op_func_count = 0
                for t_func in all_tasks[op]:
                    t0 = time.time()
                    ret = run_task(t_func, args, log_dir)
                    t1 = time.time()
                    if ret != 0:
                        print(f"[Triton CI Run Flaggems OPs][error] {t_func} failed, time cost {(t1 - t0):.2f}s!")
                        failed_task_count += 1
                        failed_op_func_count += 1
                    else:
                        print(f"[Triton CI Run Flaggems OPs][info] {t_func} passed, time cost {(t1 - t0):.2f}s!")
                        succ_task_count += 1
                        succ_op_func_count += 1
                if failed_op_func_count > 0:
                    failed_op_count += 1
                    print(f"[Triton CI Run Flaggems OPs][info] {idx}: {op} test failed, {failed_op_func_count}/{failed_op_func_count+succ_op_func_count} func failed!")
                else:
                    succ_op_count += 1
                    print(f"[Triton CI Run Flaggems OPs][info] {idx}: {op} test success!")
            else:
                print(f"[Triton CI Run Flaggems OPs][warn] {idx}: {op} not in all_tasks!")
    
        print(f"run end, succ task count: {succ_task_count}, failed task count: {failed_task_count}")
        print(f"run end, succ op count: {succ_op_count}, failed op count: {failed_op_count}")
        sys.exit(0)

    if args.func is not None:
        t_func = args.func
        t0 = time.time()
        ret = run_task(t_func, args, log_dir)
        t1 = time.time()
        if ret != 0:
            print(f"[Triton CI Run Flaggems OPs][error] {t_func} failed, time cost {(t1 - t0):.2f}s!")
        else:
            print(f"[Triton CI Run Flaggems OPs][info] {t_func} passed, time cost {(t1 - t0):.2f}s!")
        sys.exit(0)

