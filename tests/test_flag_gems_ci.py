import os,sys
import re
import json
import argparse
import queue
import time
import signal 
import subprocess
import concurrent.futures
from queue import Queue
from loguru import logger
import multiprocessing
from multiprocessing import Manager


##test_set_name必须在json文件中有定义
def read_json_ops_and_tasks(file_path, test_set_name=None):
    all_op_list = []
    test_op_list = None
    all_task_dict = {}
    hardware_bug_ops = []
    print(f"[Triton CI Run Flaggems OPs][info]{file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            all_op_list = json_data.get('all_ops', [])
            if test_set_name is not None:
                test_op_list = json_data.get(test_set_name, [])
            all_task_dict = json_data.get('all_tasks', {})
            hardware_bug_ops = json_data.get('hardware_bug_ops', [])

    except FileNotFoundError:
        print(f"[Triton CI Run Flaggems OPs][error] {file_path} not found!")
    except json.JSONDecodeError:
        print(f"[Triton CI Run Flaggems OPs][error] {file_path} data decode error!")
    except Exception as e:
        print(f"[Triton CI Run Flaggems OPs][error] {file_path} read json fail!")

    return all_op_list, test_op_list, all_task_dict, hardware_bug_ops


def check_card_status(card_id: str):
    """
    执行tsm_smi 解析NPU-Util的值
    Args:
        card_id (str): 输入card id or 返回所有卡的信息
    Returns:
        bool: True表示空闲(0%), False表示被占用
    """
    try:
        output = subprocess.check_output(
            "tsm_smi",
            shell=True,
            text=True,
            stderr=subprocess.STDOUT
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"命令执行异常: {e}")
        return False

    # logger.debug(f"output: {output}")
    lines = output.splitlines()

    for line in lines:
        # logger.debug(f"line: {line}")
        line = re.sub(r'\s{2,}', ' ', line.strip())
        line = line.replace("'", "").replace("/", "").replace("|", "").strip()
        if not line.strip() or "Card count" in line or "TSM-SMI" in line or "NPU-Util" in line:
            continue
        if "Card" in line and "Name" in line and "NPU-Util" in line:
            continue

        fields = line.split()
        # logger.debug(f"fields: {fields}")
        if len(fields) < 2:
            continue

        if str(fields[0]) != card_id and str(fields[1]) != card_id:
            continue

        for field in fields :
            if field.endswith("%"):
                util_str = field.strip("%")
                if util_str.isdigit():
                    util_value = int(util_str)
                    # logger.debug(f'util_value: {util_value}')
                    return util_value == 0

    logger.error(f"ERROR: 未找到卡号 {card_id} ")
    return False

def run_case(task, all_tasks, card_id, quick_mode):
    """
    运行各算子的test_accuracy函数,得到算子验证结果
    Args:
        task: 算子测试函数
        card_id: 当前使用的npu id
    """
    triton_cache_dir = "/tmp/triton_cache_" + card_id 
    triton_dump_dir = "/tmp/triton_dump_" + card_id 
    flaggems_cache_dir = "/tmp/flaggems_cache_" + card_id 
    env = os.environ.copy()
    env.update({
        'TXDA_VISIBLE_DEVICES': card_id,
        #'PRECISION_PRIORITY': "1",
        'TRITON_CACHE_DIR': triton_cache_dir,
        'TRITON_DUMP_PATH': triton_dump_dir,
        'FLAGGEMS_CACHE_DIR': flaggems_cache_dir,
    })
    # print(f"current case: {task}\ncard_id: {card_id}\nenviroment: {env}")
    
    #case_dir = os.path.join(CASE_WORK_DIR, task)
    CASE_WORK_DIR = os.environ.get("TRITON_WORKSPACE")
    log_dir = os.path.join(CASE_WORK_DIR, "dump/flag_gems/")
    os.makedirs(log_dir, exist_ok=True)

    op_funcs = all_tasks[task]
    file_path = os.path.dirname(os.path.abspath(__file__))
    failed_op_func_count = 0
    succ_op_func_count = 0
    failed_op_func_list = []
    print(f"[Triton CI Run Flaggems OPs][info]{task} begin test...")
    for t_func in op_funcs:
        log_file_path = os.path.join(CASE_WORK_DIR, f"dump/flag_gems/{t_func}.log")
        print(f"[Triton CI Run Flaggems OPs][info]{t_func} start test >>>>>>")
        print(f"[Triton CI Run Flaggems OPs][info]log_file_path: {log_file_path}")
        cmd = ["python3", "-m", "pytest", "-v", "-s", f"{file_path}/{t_func}", "--ref", "cpu"]
        if quick_mode:
            cmd.append("--mode")
            cmd.append("quick")
        print(f"[Triton CI Run Flaggems OPs][info]cmd: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            cwd=CASE_WORK_DIR,
            env=env,
            stdout=open(file=log_file_path, mode='w'),
            stderr=subprocess.STDOUT
        )
        
        interval = 60
        threshold = 5
        counter = 0
        prev_size = 0      
        while True:
            time.sleep(interval)  
            try:
                current_size = os.path.getsize(log_file_path)
            except FileNotFoundError:
                print(f"[Triton CI Run Flaggems OPs][error] log file {log_file_path} not exists!")
                break
            
            if current_size == prev_size:
                counter += 1
                print(f"[Triton CI Run Flaggems OPs][info]{t_func} log file size unchanged，already：{counter}/{threshold} times!")
                if counter >= threshold:
                    print(f"[Triton CI Run Flaggems OPs][error] {t_func} already stuck，timeout and stoped!")
                    # process.terminate()
                    # 发送ctrl+c信号进行资源释放
                    process.send_signal(signal.SIGINT)   
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break  
            else:
                counter = 0
                prev_size = current_size
                print(f"[Triton CI Run Flaggems OPs][info]{t_func} log is normal，current_size：{current_size} Byte.")
            
            # 判断该进程是否还在进行
            if process.poll() is not None:
                break
            
            time.sleep(10)      
        # 等待进程结束
        process.wait()
        time.sleep(10)
        if process.returncode == 0:
            print(f"[Triton CI Run Flaggems OPs][info]{t_func} test success!")
            succ_op_func_count += 1
        else:
            print(f"[Triton CI Run Flaggems OPs][error]{t_func} test failed, ret_code: {process.returncode}")
            failed_op_func_count += 1
            failed_op_func_list.append((t_func, process.returncode))
    if failed_op_func_count > 0:
        print(f"[Triton CI Run Flaggems OPs][error]{task} completed, {failed_op_func_count}/{failed_op_func_count+succ_op_func_count} func failed.")
        return failed_op_func_count, task, failed_op_func_list
    else:
        print(f"[Triton CI Run Flaggems OPs][info]{task} completed, all func success.")
        return 0, task, []

def run_stage(args):
    test_set_name = args.test_set
    quick_mode = args.quick
    json_file_dir = os.environ.get("JSON_FILE_PATH")
    json_file_path = json_file_dir + "/flag_gems_ci_ops.json"
    all_op_list, test_op_list, all_tasks, hardware_bug_ops = read_json_ops_and_tasks(json_file_path, test_set_name)
    NPU_IDS = [str(i) for i in range(args.device_count) if i not in args.skip_device]
    if test_op_list is None:
        test_op_list = all_op_list
        print(f"[Triton CI Run Flaggems OPs][warn]{test_set_name} is None, use all_op_list")

    multiprocessing.set_start_method("spawn")
    with Manager() as manager:
        task_queue = manager.Queue()
        card_queue = manager.Queue()
        pass_queue = manager.Queue()
        fail_queue = manager.Queue()

        for op in test_op_list:
            if op in all_tasks:
                task_queue.put(op)
            else:
                print(f"[Triton CI Run Flaggems OPs][warn]{op} not in all_tasks, discard!")

        for card_id in NPU_IDS:
            if check_card_status(card_id):
                card_queue.put(card_id)
            else:
                print(f"[Triton CI Run Flaggems OPs][warn]Card {card_id} is not available, discard!")
        
        process_count = min(card_queue.qsize(), task_queue.qsize())
        total_count = task_queue.qsize()
        print("[Triton CI Run Flaggems OPs][info]total task count:", total_count)
        print("[Triton CI Run Flaggems OPs][info]process count:", process_count)
        # 修改回调函数，当出现case异常时，卡则不再放回资源池中
        def callback(future, card_id):
            try:
                result, op_name, failed_op_func_list = future.result()
                if result == 0:
                    pass_queue.put(op_name)
                else:
                    fail_queue.put((op_name, failed_op_func_list))
                if (check_card_status(card_id)):
                    card_queue.put(card_id)
                    print(f"[Triton CI Run Flaggems OPs][info]Card {card_id} released and available!")
                else:
                    print(f"[Triton CI Run Flaggems OPs][warn]Card {card_id} is not available, not recycled!")
            except Exception as e:
                print(f"[Triton CI Run Flaggems OPs][error]Task Exception: {e}, Card {card_id} is not available, not recycled!")

        with concurrent.futures.ProcessPoolExecutor(max_workers=process_count) as excutor:
            futures = {}
            for _ in range(process_count):
                case_dir = task_queue.get()
                card_id = card_queue.get()
                future = excutor.submit(run_case, case_dir, all_tasks, card_id, quick_mode)
                future.add_done_callback(lambda f, cid=card_id: callback(f, cid))
                futures[future] = (case_dir, card_id)
                print(f"[Triton CI Run Flaggems OPs][info]Started: {total_count-task_queue.qsize()}: {case_dir} on card {card_id}")

            while not task_queue.empty():
                completed, _ = concurrent.futures.wait(
                    list(futures.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                # 处理已经完成的任务
                for future in completed:
                    case_dir, card_id = futures.pop(future)
                    print(f"[Triton CI Run Flaggems OPs][info]Completed: {case_dir} on card {card_id}")

                # 根据剩余卡和任务数量提交新的任务
                available = card_queue.qsize()
                to_submit = min(available, task_queue.qsize())
                for _ in range(to_submit):
                    try:
                        case_dir = task_queue.get_nowait()
                        card_id = card_queue.get()
                        new_future = excutor.submit(run_case, case_dir, all_tasks, card_id, quick_mode)
                        new_future.add_done_callback(lambda f, cid=card_id: callback(f, cid))
                        futures[new_future] = (case_dir, card_id)
                        print(f"[Triton CI Run Flaggems OPs][info]Started: {total_count-task_queue.qsize()}: {case_dir} on card {card_id}")
                    except queue.Empty:
                        break
            #处理最后完成的任务
            completed, _ = concurrent.futures.wait(
                list(futures.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in completed:
                case_dir, card_id = futures.pop(future)
                print(f"[Triton CI Run Flaggems OPs][info]Completed: {case_dir} on card {card_id}")
        print("Total ops number is: ", total_count) 
        succed_count = pass_queue.qsize()
        print("Passed ops number is: ", succed_count)
        for i in range(succed_count):
            print(f'\t{pass_queue.get()}')
        
        failed_and_not_in_decluded_ops = 0
        failed_count = fail_queue.qsize()
        print("Failed ops number is: ", failed_count)
        for i in range(failed_count):
            op_name, failed_op_func_list = fail_queue.get()
            print(f'{op_name}:')
            for (tfunc, retcode) in failed_op_func_list:
                print(f'\t{tfunc}: {retcode}')
            if op_name not in hardware_bug_ops:
                failed_and_not_in_decluded_ops += 1
        print("All tasks processed")
        if failed_and_not_in_decluded_ops/total_count < 0.05:
            return 0
        else:  
            return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlagGems OP accuracy test for Triton CI.")
    parser.add_argument("--test_set",
                        type = str,
                        help = "op set name to test, define in flag_gems_ci_ops.json, e.g. --test_set all_ops",
                        default ="all_ops")
    parser.add_argument('--quick', action='store_true', default=False,
                        help='run tests on quick mode')
    parser.add_argument("--device_count",
                        type = int,
                        help = "Maximum number of devices that can be used.",
                        default= 1)
    parser.add_argument("--skip_device",
                        type = int,
                        nargs='*',
                        help = "Devices that need to be skipped, when they are unavailable.",
                        default= [])
    args = parser.parse_args()

    print("[Triton CI Run Flaggems OPs][info]------------------all env---------------------")
    for key, value in os.environ.items():
        print(f"{key}={value}")
    print("[Triton CI Run Flaggems OPs][info]----------------------------------------------")
    start_time = time.time()
    exit_code = run_stage(args)
    end_time = time.time()
    print(f"[Triton CI Run Flaggems OPs][info]time cost: {(end_time - start_time):.2f}s")
    if exit_code is not None:
        if exit_code ==0:
            sys.exit(0)
        else:
            sys.exit(-1)
    else:
        sys.exit(-1)
