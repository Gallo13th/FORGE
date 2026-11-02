import os

with open('./for_cd_hit.fa','w') as f:
    for seq_file in os.listdir('./seq'):
        seq_fasta = open('./seq/'+seq_file,'r').readlines()
        f.write(''.join(seq_fasta))
        if not seq_fasta[-1].endswith('\n'):
            f.write('\n')