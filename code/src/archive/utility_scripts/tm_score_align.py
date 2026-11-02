import os
import sys
import shutil
import tqdm
import numpy as np
import subprocess
from functools import partial
from multiprocessing import Pool, Manager

# TMscore 命令模板（请确保路径正确）
TMSCORE_CMD = "./TMscore {pdba} {pdbb}"

# PDB 文件目录
QUERY_PDB_DIR = "./restored_pdb"
TARGET_PDB_DIR = "./C3_pdb"

def main():
    # 获取所有 PDB 文件名（按顺序排列）
    query_pdb_ids = sorted(os.listdir(QUERY_PDB_DIR))
    target_pdb_ids = sorted(os.listdir(TARGET_PDB_DIR))
    n_query_pdbs = len(query_pdb_ids)
    n_target_pdbs = len(target_pdb_ids)

    # 完整路径转换为绝对路径
    query_pdb_paths = [os.path.join(QUERY_PDB_DIR, pdb_id) for pdb_id in query_pdb_ids]
    target_pdb_paths = [os.path.join(TARGET_PDB_DIR, pdb_id) for pdb_id in target_pdb_ids]

    # 存储代表结构的索引
    representatives = []

    # 初始化保留结构列表
    print("Starting representative selection...")
    with open('./tmscore_result_between_train_test2.txt', 'w') as tmscore_result:
        for idx in tqdm.tqdm(range(len(query_pdb_paths)), desc="Selecting Representatives"):
            current_query_path = query_pdb_paths[idx]
            
            # 和已有代表结构比对
            results = parallel_tmscore_computation(current_query_path, target_pdb_paths)
            
            is_similar = False
            for qry_path, tar_path, score in results:
                if score > 0.0:
                    tmscore_result.write(f"{qry_path}\t{tar_path}\t{score}\n")
                if score > 0.9:
                    is_similar = True
                    break
            
            if not is_similar:
                representatives.append(idx)

def compute_tmscore(stop_event, pdba, pdbb):
    # 检查是否应该停止
    if stop_event.is_set():
        return (pdba, pdbb, 0.0)
    
    try:
        len_a = len(open(pdba).readlines())
        len_b = len(open(pdbb).readlines())
        if abs(len_a - len_b) >= 1:
            return (pdba, pdbb, 0.0)
        
        cmd = TMSCORE_CMD.format(pdba=pdba, pdbb=pdbb)
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        for line in result.splitlines():
            if "TM-score    =" in line:
                tmscore = float(line.split("=")[1].strip().split()[0][:5])
                if tmscore > 0.6:
                    # 设置停止事件
                    stop_event.set()
                return (pdba, pdbb, tmscore)
    except Exception as e:
        print(f"Error comparing structure {pdba} and {pdbb}: {e}")
    return (pdba, pdbb, 0.0)

def parallel_tmscore_computation(query_path, target_paths):
    # 创建多进程管理器
    with Manager() as manager:
        # 创建共享的停止事件
        stop_event = manager.Event()
        
        # 创建部分函数，固定stop_event和query_path参数
        func = partial(compute_tmscore, stop_event, query_path)
        
        results = []
        # 使用进程池
        with Pool() as pool:
            # 使用imap_unordered以便更快获取结果
            try:
                for result in pool.imap_unordered(func, target_paths):
                    results.append(result)
                    if stop_event.is_set():
                        # 提前终止
                        pool.terminate()
                        break
            except Exception as e:
                print(f"Error in parallel computation: {e}")
                pool.terminate()
        
        return results

if __name__ == '__main__':
    main()