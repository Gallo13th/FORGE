import os
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# from run_xgb_model_v2 import transform_coord_seq, read_fasta_biopython, build_graph
from generate_torch_features_v0 import build_graph


def process_single_sample(args, coords_root_dir, seq_root_dir, output_dir, save_restored=False, save_error_metrics=True):
    """
    处理单个样本
    """
    idx, row = args
    pdb_id = row['pdb_id'].upper()
    chain : str = row['chain']
    if chain.isupper():
        chain = 'H_' + chain
    elif chain.islower():
        chain = 'L_' + chain
    else:
        chain = 'X_' + chain
    pdb_head = f"{pdb_id}_{chain}"
    try:
        # 构建文件路径
        coord_file = os.path.join(coords_root_dir, f"{pdb_head}.npy")
        seq_file = os.path.join(seq_root_dir, f"{pdb_head}.fasta")
        
        # 检查文件是否存在
        if not os.path.exists(coord_file):
            print(f"❌ Coordinate file not found: {coord_file}")
            print(coord_file,file=open('./not_found_coord.txt','a'))
            raise FileNotFoundError(f"Coordinate file not found: {coord_file}")

        # 加载数据
        test_coords = np.load(coord_file)
        if test_coords.shape[0] > 500:
            raise NotImplementedError("sequence is too long, skip")
        if test_coords.shape[0] < 5:
            raise NotImplementedError("sequence is too short, skip")
        # 构建图数据
        processed_data = build_graph(np.nan_to_num(test_coords[:, :7]))
        
        
        # 准备保存数据
        save_dict = {
            key: value.numpy() if isinstance(value, torch.Tensor) else value
            for key, value in processed_data.items()
        }
        np.savez_compressed(
            os.path.join(output_dir, 'proccessed', f"{pdb_head}.npz"),
            **save_dict
        )
        return {
            'pdb_id': pdb_id,
            'chain': chain,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ Error processing {pdb_head}: {str(e)}")
        return {
            'pdb_id': pdb_id,
            'chain': chain,
            'status': 'error',
            'error': str(e)
        }


def batch_process_quantization(rfam_df, coords_root_dir, seq_root_dir, output_dir, 
                             max_workers=4, save_restored=False, save_error_metrics=True):
    """
    批量处理量化
    """
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'proccessed'), exist_ok=True)
    if save_restored:
        os.makedirs(os.path.join(output_dir, 'restored'), exist_ok=True)
    if save_error_metrics:
        os.makedirs(os.path.join(output_dir, 'errors'), exist_ok=True)
    
    print(f"📊 Processing {len(rfam_df)} samples...")
    print(f"📂 Output directory: {output_dir}")
    print(f"👥 Max workers: {max_workers}")
    
    # 准备参数
    args_list = [(idx, row) for idx, row in rfam_df.iterrows()]
    # 已经处理好的样本跳过
    processed_files = set(os.listdir(os.path.join(output_dir, 'proccessed')))
    args_list = [args for args in args_list if f"{args[1]['pdb_id'].upper()}_{('H_' if args[1]['chain'].isupper() else 'L_' if args[1]['chain'].islower() else 'X_')}{args[1]['chain']}.npz" not in processed_files]
    # 对已经处理的，但是文件读取错误的，重新处理
    error_files = []
    error_handles = []
    for proc_file in processed_files:
        try:
            np.load(os.path.join(output_dir, 'proccessed', proc_file),allow_pickle=True)
        except:
            pdb_id, chain = proc_file.split('_')[0], proc_file.split('_')[2]
            error_files.append(proc_file)
            error_handles.append((rfam_df[(rfam_df['pdb_id'].str.upper() == pdb_id) & (rfam_df['chain'] == chain)].index[0], rfam_df[(rfam_df['pdb_id'].str.upper() == pdb_id) & (rfam_df['chain'] == chain)].iloc[0]))
    print(error_handles)
    args_list.extend(error_handles)
    
    
    # 使用partial固定部分参数
    process_func = partial(process_single_sample, 
                          coords_root_dir=coords_root_dir,
                          seq_root_dir=seq_root_dir,
                          output_dir=output_dir,
                          save_restored=save_restored,
                          save_error_metrics=save_error_metrics)
    
    results = []
    successful_count = 0
    error_count = 0
    
    # 使用tqdm显示进度
    with tqdm(total=len(args_list), desc="Processing samples") as pbar:
        if max_workers > 1:
            # 多进程处理
            with mp.Pool(processes=max_workers) as pool:
                for result in pool.imap_unordered(process_func, args_list):
                    results.append(result)
                    if result and result['status'] == 'success':
                        successful_count += 1
                    else:
                        error_count += 1
                    pbar.update(1)
                    pbar.set_postfix(success=successful_count, error=error_count)
        else:
            # 单进程处理
            for args in args_list:
                result = process_func(args)
                results.append(result)
                if result and result['status'] == 'success':
                    successful_count += 1
                else:
                    error_count += 1
                pbar.update(1)
                pbar.set_postfix(success=successful_count, error=error_count)
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, 'processing_summary.json')
    summary = {
        'total_samples': len(rfam_df),
        'successful_samples': successful_count,
        'error_samples': error_count,
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Batch processing completed!")
    print(f"   Total samples: {len(rfam_df)}")
    print(f"   Successful: {successful_count}")
    print(f"   Errors: {error_count}")
    print(f"   Summary saved to: {summary_file}")
    
    return results

if __name__ == '__main__':
    # 配置路径
    coords_root_dir = "./coords/"
    seq_root_dir = "./seq/"
    output_dir = "./rfam_quantization"
    os.makedirs(output_dir, exist_ok=True)
    # 读取数据
    rfam_df = pd.read_csv('./rfam_pdb_labeled.csv', sep='\t')
    
    # 如果只想处理前N个样本进行测试
    # rfam_df = rfam_df.head(10)  # 只处理前10个样本
    
    # 批量处理
    results = batch_process_quantization(
        rfam_df=rfam_df,
        coords_root_dir=coords_root_dir,
        seq_root_dir=seq_root_dir,
        output_dir=output_dir,
        max_workers=1,  # 根据你的CPU核心数调整
        save_restored=False,  # 是否保存还原数据（占用空间大）
        save_error_metrics=True  # 是否保存误差指标
    )
    
    # test_seq_file = "./seq/1A1T_H_B.fasta"
    # test_coord_file = "./coords/1A1T_H_B.npy"
    
    # coords = np.load(test_coord_file)
    # processed_data = build_graph(np.nan_to_num(coords[:, :7]))
    # print(processed_data['x'].shape, processed_data['edge_attr'].shape)
    # print(processed_data['node_feat_name'])
    # print(processed_data['edge_feat_name'])