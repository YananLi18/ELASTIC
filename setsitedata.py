
import torch
import csv
# CPU:25G BD:8.6G MTR:37G BW 1.5G
import os
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='generate site dir for whole train')

parser.add_argument('--root_path', type=str, default='/data/zph/HGCN/data/', help='root path')
parser.add_argument('--data', type=str, default='All_full', help='data path')

parser.add_argument('--freq', type=str, default='15min', help='frequency')
parser.add_argument("--output_dir", type=str, default="/data/zph/HGCN/data/site/", help="Output directory.")

args = parser.parse_args()
# 输入输出目录
data_path = args.root_path + args.data + '/' + args.freq + '/'
folder_path = args.output_dir + args.freq + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 初始化vm和site的目录
ins = pd.read_csv(args.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores']) #所有vm/site数据
uuid_to_region = ins.set_index('uuid')['ens_region_id'].to_dict()
need_vm = pd.read_csv(data_path+'vmlist.csv')                            # 保证vmlist就是cpu 和bw的vm，完全相同
need_vm = need_vm['vm'].values                                           # 转换为列表
uuid_to_region = {k:v for k,v in uuid_to_region.items() if k in need_vm} # 只提取数据里有的vm和对应的站点

# get site list as vmlist in dir of /site/ 
regions = pd.DataFrame(uuid_to_region.values(),columns = ['ens_region_id'])
regions.drop_duplicates(inplace=True)
regions.to_csv(folder_path + 'vmlist.csv', index=False)

# get site cpu

cpu = pd.read_csv(data_path + 'cpu_rate.csv')
print('start get cpu...')
for region in regions['ens_region_id']:

    region_uuids = [uuid for uuid, region_id in uuid_to_region.items() if region_id == region]
    for uuid in region_uuids:#每个vm*芯片数
        cores = ins[ins['uuid']==uuid]['cores'].values[0]
        cpu[uuid] = cpu[uuid]*cores
    cpu[region] = cpu[region_uuids].sum(axis=1)#每个site是所有旗下vm的累加
    
    cpu.drop(columns=region_uuids, inplace=True)
cpu.to_csv(folder_path + 'cpu_rate.csv', index=False)
print('cpu data done...')

# get site bw
bw = pd.read_csv(data_path + 'up_bw.csv')
print('start get bw...')
for region in regions['ens_region_id']:
    region_uuids = [uuid for uuid, region_id in uuid_to_region.items() if region_id == region]
    bw[region] = bw[region_uuids].sum(axis=1)#每个site是所有旗下vm的累加
    bw[region] = bw[region] / 1e6  # bps -> Mbps
    bw.drop(columns=region_uuids, inplace=True)

bw.to_csv(folder_path + 'up_bw.csv', index=False)
print('bw data done...')
