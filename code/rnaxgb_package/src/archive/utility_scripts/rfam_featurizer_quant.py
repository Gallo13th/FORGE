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


def quantize_tensor(tensor, num_bits=8):
    """
    对张量进行线性量化，并返回量化后的张量、最小值、最大值
    """
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    min_val = tensor.min()
    max_val = tensor.max()

    scale = (max_val - min_val) / qmax
    zero_point = qmin - min_val / scale

    # Quantize
    quantized = torch.round((tensor - min_val) / scale).clamp(qmin, qmax)
    return quantized.to(torch.uint8), min_val, max_val, scale


def dequantize_tensor(quantized_tensor, min_val, max_val, num_bits=8):
    """
    将量化后的张量还原为原始范围
    """
    qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / qmax
    dequantized = quantized_tensor.float() * scale + min_val
    return dequantized


def calculate_error_metrics(original, restored):
    """
    计算误差指标
    """
    original = original.flatten()
    restored = restored.flatten()
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(original - restored))
    
    # Mean Squared Error
    mse = torch.mean((original - restored) ** 2)
    
    # Root Mean Squared Error
    rmse = torch.sqrt(mse)
    
    # Maximum Absolute Error
    max_error = torch.max(torch.abs(original - restored))
    
    # Signal-to-Noise Ratio (SNR)
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - restored) ** 2)
    if noise_power > 0:
        snr = 10 * torch.log10(signal_power / noise_power)
    else:
        snr = torch.tensor(float('inf'))
    
    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item(),
        'Max_Error': max_error.item(),
        'SNR': snr.item()
    }


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
    if pdb_id == '3SKT':
        print("================")
        print(pdb_head)
    try:
        # 构建文件路径
        coord_file = os.path.join(coords_root_dir, f"{pdb_head}.npy")
        seq_file = os.path.join(seq_root_dir, f"{pdb_head}.fasta")
        
        # 检查文件是否存在
        if not os.path.exists(coord_file):
            print(f"❌ Coordinate file not found: {coord_file}")
            print(coord_file+'\n',file=open('./not_found_coord.txt','a'))
            return None

        # 加载数据
        test_coords = np.load(coord_file)
        if test_coords.shape[0] > 500:
            raise NotImplementedError("sequence is too long, skip")
        if test_coords.shape[0] < 5:
            raise NotImplementedError("sequence is too short, skip")
        # 构建图数据
        processed_data = build_graph(np.nan_to_num(test_coords[:, :7]))
        processed_data['x_log_flag'] = torch.zeros(processed_data['x'].shape[1], dtype=torch.bool)
        processed_data['edge_attr_log_flag'] = torch.zeros(processed_data['edge_attr'].shape[1], dtype=torch.bool)
        
        # 保存原始数据用于误差计算
        original_x = processed_data['x'].clone()
        original_edge_attr = processed_data['edge_attr'].clone()
        
        # Quantization storage
        x_quant_params = {}
        edge_attr_quant_params = {}
        
        # Process node features
        for feat_idx, feat_name in enumerate(processed_data['node_feat_name']):
            x = processed_data['x'][:, feat_idx]
            if torch.max(torch.abs(x)) > 10:
                x = torch.sign(x) * torch.log(torch.abs(x) + 1)
                processed_data['x_log_flag'][feat_idx] = True
                processed_data['x'][:, feat_idx] = x

            # Quantize node feature
            quantized_x, min_val, max_val, scale = quantize_tensor(x, num_bits=8)
            x_quant_params[feat_idx] = {'min': min_val.item(), 'max': max_val.item(), 'scale': scale.item()}
            processed_data['x'][:, feat_idx] = quantized_x.float()
        
        # Process edge attributes
        for feat_idx in range(processed_data['edge_attr'].shape[1]):
            x = processed_data['edge_attr'][:, feat_idx]
            if torch.max(torch.abs(x)) > 10:
                x = torch.sign(x) * torch.log(torch.abs(x) + 1)
                processed_data['edge_attr_log_flag'][feat_idx] = True
                processed_data['edge_attr'][:, feat_idx] = x

            # Quantize edge attribute
            quantized_x, min_val, max_val, scale = quantize_tensor(x, num_bits=8)
            edge_attr_quant_params[feat_idx] = {'min': min_val.item(), 'max': max_val.item(), 'scale': scale.item()}
            processed_data['edge_attr'][:, feat_idx] = quantized_x.float()
        
        # 准备保存数据
        save_dict = {
            key: value.numpy() if isinstance(value, torch.Tensor) else value
            for key, value in processed_data.items()
        }
        save_dict['x_quant_params'] = x_quant_params
        save_dict['edge_attr_quant_params'] = edge_attr_quant_params
        
        # 保存量化后的数据
        quant_file = os.path.join(output_dir, 'quantized', f"{pdb_head}_quant.npz")
        np.savez_compressed(quant_file, **save_dict)
        
        error_metrics = None
        if save_error_metrics or save_restored:
            # 还原数据计算误差
            restored_x = torch.zeros_like(original_x)
            restored_edge_attr = torch.zeros_like(original_edge_attr)
            
            # Restore node features
            for feat_idx in range(processed_data['x'].shape[1]):
                quantized_x = torch.tensor(save_dict['x'][:, feat_idx])
                params = x_quant_params[feat_idx]
                restored_x[:, feat_idx] = dequantize_tensor(quantized_x, params['min'], params['max'], num_bits=8)
                if processed_data['x_log_flag'][feat_idx]:
                    restored_x[:, feat_idx] = torch.sign(restored_x[:, feat_idx]) * (torch.exp(torch.abs(restored_x[:, feat_idx])) - 1)
            
            # Restore edge attributes
            for feat_idx in range(processed_data['edge_attr'].shape[1]):
                quantized_x = torch.tensor(save_dict['edge_attr'][:, feat_idx])
                params = edge_attr_quant_params[feat_idx]
                restored_edge_attr[:, feat_idx] = dequantize_tensor(quantized_x, params['min'], params['max'], num_bits=8)
                if processed_data['edge_attr_log_flag'][feat_idx]:
                    restored_edge_attr[:, feat_idx] = torch.sign(restored_edge_attr[:, feat_idx]) * (torch.exp(torch.abs(restored_edge_attr[:, feat_idx])) - 1)
            
            if save_restored:
                # 保存还原数据
                restored_dict = {
                    'restored_x': restored_x.numpy(),
                    'restored_edge_attr': restored_edge_attr.numpy()
                }
                restored_file = os.path.join(output_dir, 'restored', f"{pdb_head}_restored.npz")
                np.savez_compressed(restored_file, **restored_dict)
            
            if save_error_metrics:
                # 计算误差指标
                node_errors = calculate_error_metrics(original_x, restored_x)
                edge_errors = calculate_error_metrics(original_edge_attr, restored_edge_attr)
                all_original = torch.cat([original_x.flatten(), original_edge_attr.flatten()])
                all_restored = torch.cat([restored_x.flatten(), restored_edge_attr.flatten()])
                overall_errors = calculate_error_metrics(all_original, all_restored)
                
                error_metrics = {
                    'pdb_id': pdb_id,
                    'chain': chain,
                    'node_features': node_errors,
                    'edge_attributes': edge_errors,
                    'overall': overall_errors
                }
                
                # 保存单个样本的误差指标
                error_file = os.path.join(output_dir, 'errors', f"{pdb_head}_error.json")
                with open(error_file, 'w') as f:
                    json.dump(error_metrics, f, indent=2)
        return {
            'pdb_id': pdb_id,
            'chain': chain,
            'status': 'success',
            'x_shape': original_x.shape,
            'edge_attr_shape': original_edge_attr.shape,
            'x_log_features': processed_data['x_log_flag'].sum().item(),
            'edge_attr_log_features': processed_data['edge_attr_log_flag'].sum().item(),
            'error_metrics': error_metrics
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
    os.makedirs(os.path.join(output_dir, 'quantized'), exist_ok=True)
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
    processed_files = set(os.listdir(os.path.join(output_dir, 'quantized')))
    args_list = [args for args in args_list if f"{args[1]['pdb_id'].upper()}_{('H_' if args[1]['chain'].isupper() else 'L_' if args[1]['chain'].islower() else 'X_')}{args[1]['chain']}_quant.npz" not in processed_files]
    # 对已经处理的，但是文件读取错误的，重新处理
    error_files = []
    error_handles = []
    for proc_file in processed_files:
        try:
            np.load(os.path.join(output_dir, 'quantized', proc_file),allow_pickle=True)
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
    
    # 如果保存了误差指标，生成汇总报告
    if save_error_metrics:
        generate_error_summary(output_dir)
    
    return results


def generate_error_summary(output_dir):
    """
    生成误差指标汇总报告
    """
    error_dir = os.path.join(output_dir, 'errors')
    if not os.path.exists(error_dir):
        return
    
    error_files = [f for f in os.listdir(error_dir) if f.endswith('_error.json')]
    
    if not error_files:
        return
    
    print("📊 Generating error summary...")
    
    all_node_errors = []
    all_edge_errors = []
    all_overall_errors = []
    
    for error_file in error_files:
        try:
            with open(os.path.join(error_dir, error_file), 'r') as f:
                error_data = json.load(f)
                all_node_errors.append(error_data['node_features'])
                all_edge_errors.append(error_data['edge_attributes'])
                all_overall_errors.append(error_data['overall'])
        except Exception as e:
            print(f"❌ Error reading {error_file}: {str(e)}")
    
    if all_overall_errors:
        # 计算平均误差指标
        metrics = ['MAE', 'MSE', 'RMSE', 'Max_Error', 'SNR']
        avg_summary = {}
        
        for metric in metrics:
            node_avg = np.mean([err[metric] for err in all_node_errors])
            edge_avg = np.mean([err[metric] for err in all_edge_errors])
            overall_avg = np.mean([err[metric] for err in all_overall_errors])
            
            avg_summary[metric] = {
                'node_features': node_avg,
                'edge_attributes': edge_avg,
                'overall': overall_avg
            }
        
        summary_file = os.path.join(output_dir, 'error_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(avg_summary, f, indent=2)
        
        print(f"📈 Error summary saved to: {summary_file}")
        print("📊 Average Error Metrics:")
        for metric in metrics:
            print(f"   {metric}:")
            print(f"     Node Features: {avg_summary[metric]['node_features']:.6f}")
            print(f"     Edge Attributes: {avg_summary[metric]['edge_attributes']:.6f}")
            print(f"     Overall: {avg_summary[metric]['overall']:.6f}")


if __name__ == '__main__':
    # 配置路径
    coords_root_dir = "./coords/"
    seq_root_dir = "./seq/"
    output_dir = "./rfam_quantization"
    os.makedirs(output_dir, exist_ok=True)
    # 读取数据
    # rfam_df = pd.read_csv('./rfam_pdb_labeled.csv', sep='\t')
    
    # 如果只想处理前N个样本进行测试
    # rfam_df = rfam_df.head(10)  # 只处理前10个样本
    
    # 批量处理
    # results = batch_process_quantization(
    #     rfam_df=rfam_df,
    #     coords_root_dir=coords_root_dir,
    #     seq_root_dir=seq_root_dir,
    #     output_dir=output_dir,
    #     max_workers=1,  # 根据你的CPU核心数调整
    #     save_restored=False,  # 是否保存还原数据（占用空间大）
    #     save_error_metrics=True  # 是否保存误差指标
    # )
    
    test_seq_file = "./seq/1A1T_H_B.fasta"
    test_coord_file = "./coords/1A1T_H_B.npy"
    
    coords = np.load(test_coord_file)
    processed_data = build_graph(np.nan_to_num(coords[:, :7]))
    print(processed_data['x'].shape, processed_data['edge_attr'].shape)
    print(processed_data['node_feat_name'])
    print(processed_data['edge_feat_name'])