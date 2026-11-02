import numpy as np

# def load_compressed_data(file_path):
#     """
#     加载压缩文件并解析为字典 {filename: coords_data}
#     Args:
#         file_path: 压缩文件路径
#     Returns:
#         dict: {文件名: 压缩数据列表}
#     """
#     with open(file_path, 'r') as f:
#         lines = [line.strip() for line in f.readlines() if line.strip() != '']
    
#     data_dict = {}
#     current_file = None
#     for line in lines:
#         if line.endswith('.npy'):  # 文件名行
#             current_file = line
#             data_dict[current_file] = []
#         else:  # 坐标数据行
#             if current_file is None:
#                 raise ValueError("数据格式错误：坐标数据前缺少文件名")
#             data_dict[current_file].extend(line.split())
#     return data_dict

# def restore_coordinates(compressed_data):
#     """
#     从压缩数据还原原始数组 [N,7,3]（含NaN）
#     Args:
#         compressed_data: 压缩数据列表（含"NaN"字符串）
#     Returns:
#         tuple: (first_point, restored_data)
#         - first_point: 平移基准点 [x,y,z]
#         - restored_data: 形状为 [N,7,3] 的数组，含NaN
#     """
#     if not compressed_data:
#         return None, np.empty((0, 7, 3), dtype=np.float32)
    
#     # 查找第一个非NaN的点作为平移基准
#     first_point = None
#     for i in range(0, len(compressed_data), 3):
#         x, y, z = compressed_data[i], compressed_data[i+1], compressed_data[i+2]
#         if x != "NaN":
#             first_point = np.array([float(x), float(y), float(z)])
#             break
    
#     if first_point is None:  # 全部为NaN的情况
#         first_point = np.zeros(3)
#         N = len(compressed_data) // (7 * 3)
#         return first_point, np.full((N, 7, 3), np.nan)
    
#     # 还原数据
#     total_values = len(compressed_data)
#     N = total_values // (7 * 3)
#     arr = np.full((N, 7, 3), np.nan, dtype=np.float32)
    
#     idx = 0
#     for i in range(N):
#         for j in range(7):
#             x = compressed_data[idx]
#             y = compressed_data[idx+1]
#             z = compressed_data[idx+2]
#             idx += 3
#             if x != "NaN":
#                 arr[i, j] = [float(x), float(y), float(z)] + first_point
#     return first_point, arr

def generate_minimal_rna_pdb(bases):
    """
    根据RNA碱基列表生成一个最小化的PDB字符串，用于结构比对。
    
    参数:
        bases (list): 包含每个碱基信息的字典列表，格式如下：
            {
                'type': 'A',
                'number': 1,
                'C3\''(x, y, z),
            }
    返回:
        str: PDB 格式的字符串
    """
    pdb_lines = []

    atom_counter = 1

    for base in bases:
        base_type = base['type']
        res_seq = base['number']
        chain_id = "A"  # 默认链ID，可修改

        c3_prime_coords = base.get("C3'")

        # 构建 ATOM 行
        for name, coords in [("C3'", c3_prime_coords)]:
            x, y, z = coords
            line = (
                f"ATOM  {atom_counter:>5}  "
                f"{name:<4}"
                f"{base_type:>3} {chain_id}{res_seq:>4}"
                f"    "
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}"
                f"  1.00  0.00      {chain_id}   "
                f"{name[0]:>2}"
            )
            pdb_lines.append(line)
            atom_counter += 1

    return "\n".join(pdb_lines)
import os
from tqdm import tqdm
# 示例用法
if __name__ == "__main__":
    
    # 逐个文件还原
    for filename in tqdm(os.listdir('./C3_coords')):
        restored_data = np.load(f'./C3_coords/{filename}')
        if os.path.exists('./C3_pdb/'+filename.replace('.npy','.pdb')):
            continue
        bases = [
            {
                'type' : 'X',
                'number' : i,
                'C3\'': restored_data[i,:]
            } for i in range(len(restored_data))
        ]
        
        pdb_lines = generate_minimal_rna_pdb(bases)
        with open('./C3_pdb/'+filename.replace('.npy','.pdb'),'w') as f:
            f.write(pdb_lines)
        # print(f"\n文件名: {filename}")
        # print(f"平移基准点: {first_point}")
        # print(f"还原后数组形状: {restored_data.shape}")
        # print("第一帧数据:")
        # print(restored_data[0])