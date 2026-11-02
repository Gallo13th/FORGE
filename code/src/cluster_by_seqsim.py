import random
from collections import defaultdict
import pandas as pd
import os

def parse_fasta(file_path):
    clusters = defaultdict(list)
    current_cluster = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_cluster = line.split()[1]
            elif line:
                clusters[current_cluster].append(line.split('...')[0].split('>')[1])
    
    return clusters

def main():
    input_file = 'ndb_cd_hit_0.8.clstr'  # 替换为你的输入文件路径
    output_dir = 'split_data'       # 输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析FASTA文件
    clusters = parse_fasta(input_file)
    
    # 保存数据集
    data = []
    for clst_idx, clst_data in clusters.items():
        data.extend(
            [
                {
                'pdb_id' : pdb_id,
                'cluster_idx' : clst_idx,
                }
            for pdb_id in clst_data] 
        )
    pd.DataFrame(data).to_csv('cluster_by_seqsim_0.8.tsv',sep='\t',index=False)

if __name__ == '__main__':
    main()