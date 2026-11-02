import pandas as pd

# results = open('./tmscore_result_between_train_test.txt').readlines()
split_id = lambda x: x.split('\\')[-1][:-4]

df = pd.read_csv('./tmscore_result_between_train_test2.txt',sep='\t',names=['TEST','TRAIN','TMSCORE'])
df['TEST'] = df['TEST'].apply(split_id)
df['TRAIN'] = df['TRAIN'].apply(split_id)
print(df)
print(len(df[df['TMSCORE'] >= 0.8]['TRAIN'].unique()))
print(len(df[df['TMSCORE'] >= 0.8]['TEST'].unique()))

df[df['TMSCORE'] >= 0.8].to_csv('test_id.csv')
with open('./all_seq_to_cd_hit.fa','w') as f:
    for idx, (test, train, tmscore) in df[df['TMSCORE'] >= 0.8].iterrows():
        f.write(''.join(open(f'./seq/{train}.fasta').readlines()))
        f.write('\n')
    

# import sys
# import re

# def parse_clusters(file_path):
#     cluster_data = {}
#     cluster_data_rev = {}
#     current_cluster = None

#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             # 匹配 Cluster 标题
#             cluster_match = re.match(r'>Cluster\s+(\d+)', line)
#             if cluster_match:
#                 current_cluster = int(cluster_match.group(1))
#                 cluster_data[current_cluster] = []
#                 continue

#             # 匹配 PDB ID 行
#             pdb_match = re.match(r'\d+\s+\d+nt,\s+>(\w+_\d+_[^\.]+)', line)
#             if pdb_match and current_cluster is not None:
#                 pdb_id = pdb_match.group(1)
#                 cluster_data_rev[pdb_id] = current_cluster
#                 cluster_data[current_cluster].append(pdb_id)

#     return cluster_data,cluster_data_rev

# file_path = 'output.fasta.clstr'
# cluster_data,cluster_data_rev = parse_clusters(file_path)
# # print(cluster_data)
# # print(cluster_data_rev)

# must_train_pdb_id = df[df['TMSCORE'] >= 0.5]['TRAIN'].unique()
# must_train_cluster = [cluster_data_rev.get(pdb_id,pdb_id) for pdb_id in must_train_pdb_id]
# train_pdb_id = []
# test_pdb_id = []
# valid_pdb_id = []
# visited_cluster = set(must_train_cluster)
# import os
# import random
# for pdb_id_npy in os.listdir('RNAdesignv1/train/coords'):
#     pdb_id = pdb_id_npy[:-4]
#     if pdb_id in must_train_pdb_id:
#         train_pdb_id.append(pdb_id)
#         continue
#     if pdb_id in cluster_data_rev:
#         if cluster_data_rev[pdb_id] in visited_cluster:
#             if random.random() > 0.5:
#                 valid_pdb_id.append(pdb_id)
#                 continue
#             else:
#                 train_pdb_id.append(pdb_id)
#                 continue
#         else:
#             train_pdb_id.append(pdb_id)
#     else:
#         train_pdb_id.append(pdb_id)
# print(len(train_pdb_id))
# print(len(valid_pdb_id))

# train_pdb_id_df = pd.DataFrame(
#     [{'pdb_id':pdb_id,'split':'TRAIN'} for pdb_id in train_pdb_id] + [{'pdb_id':pdb_id,'split':'VALID'} for pdb_id in valid_pdb_id]
# )
# print(train_pdb_id_df)
# train_pdb_id_df.to_csv('split_data.csv',sep='\t',index=False)
# print(must_train_cluster)

# # 打印结果
# for cluster_id, pdb_ids in sorted(cluster_data.items()):
#     print(f"Cluster {cluster_id}:")
#     for pdb_id in pdb_ids:
#         print(f"  {pdb_id}")
