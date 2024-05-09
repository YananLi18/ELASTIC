#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：PycharmProjects 
@File ：dataset.py
@Author ：Yanan Li
@Date ：2021-09-01 19:45 
@Version :   1.0
@Contact :   YaNanLi@bupt.edu.cn
@Desc    :   Generate experimental data sets
'''
import os
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--region', type=str, default='CC', help='which region (CC, EC, NC, NW, SC, SW)')
parser.add_argument('--gra', type=str, default='15min', help='Granularity of time series (5, 10, 15, 30, 1h)')
parser.add_argument("--output_dir", type=str, default="/data2/lyn/www23/data/", help="Output directory.")
args = parser.parse_args()

dicT = {"5min": 8640, '10min': 4320, '15min': 2880, '30min': 1440, 'H': 720}
limits = dicT[args.gra]

if args.region == 'CC':
    region_list = ['wuhan-telecom', 'zhengzhou-telecom', 'nanchang-telecom', 'jingzhou-telecom',
                   'changsha-telecom', 'wuhan-telecom-3', 'wuhan-telecom-4']
elif args.region == 'EC':
    region_list = ['suzhou-telecom-2', 'qingdao-telecom', 'shanghai-telecom',
                   'suzhou-telecom-3', 'shanghai-telecom-2', 'shanghai-telecom-3', 'suzhou-telecom',
                   'shanghai-telecom-5', 'wenzhou-telecom', 'wuhu-telecom',
                   'shanghai-telecom-4', 'wuxi-telecom-2', 'hangzhou-telecom', 'shanghai-telecom-6']
elif args.region == 'NC':
    region_list = ['tianjin-telecom', 'beijing-telecom', 'shijiazhuang-telecom',
                   'shijiazhuang-telecom-2']
elif args.region == 'NW':
    region_list = ['xian-telecom', 'lanzhou-telecom', 'xian-telecom-2', 'xining-telecom']
elif args.region == 'SC':
    region_list = ['guangzhou-telecom', 'fuzhou-telecom', 'dongguan-telecom',
                   'dongguan-telecom-2', 'foshan-telecom', 'foshan-telecom-2',
                   'xiamen-telecom', 'shenzhen-telecom', 'haikou-telecom', 'nanning-telecom']
elif args.region == 'SW':
    region_list = ['kunming-telecom', 'chengdu-telecom-3', 'chengdu-telecom',
                   'chongqing-telecom', 'chengdu-telecom-2', 'chongqing-telecom-2', 'chengdu-telecom-4']
elif args.region == 'NE':
    region_list = ['dalian-telecom-2', 'haerbin-telecom', 'tonghua-telecom']


root_path = '/data/ali_ENS/'

bw = pd.read_csv(root_path+'T_INSTANCE_BANDWIDTH.csv')
bw.drop(['pub_down_flow', 'pub_up_flow', 'pri_down_flow', 'pri_up_flow',
        'create_time', 'pri_down_bw', 'pri_up_bw'], axis=1, inplace=True)
rebw = bw[bw['region_id'].isin(region_list)]
rebw['date'] = pd.to_datetime(rebw['report_ts'], unit='s')
rebw['date'] = pd.DatetimeIndex(rebw['date']) + timedelta(hours=8)
rebw.drop(["report_ts"], axis=1, inplace=True)

region_ins_u = rebw.groupby(['ins_id', 'date'])[['pub_up_bw']].mean()#一行只有三个数据，insid，date，bw均值(去重)

ins = pd.read_csv(root_path+'e_vm_instance.csv', usecols=['uuid', 'ens_region_id'])
ins_re = ins[ins['ens_region_id'].isin(region_list)]
ins_r_l = ins_re['uuid'].unique().tolist()
vmlist_u = []
N = True
for i, vm in enumerate(region_ins_u.index.levels[0]):#多级索引中的第一层ins_id。不重复地枚举 ins_id。
    if vm not in ins_r_l:
        continue
    region_pm = region_ins_u.loc[vm].resample(args.gra).mean()#vm虚拟机在所有不同日期下的数据。
    region_pm.rename(columns={'pub_up_bw': vm}, inplace=True)
    region_pm.dropna(inplace=True)
    if len(region_pm) < limits:
        continue
    if N:
        all_ins_u = region_pm
    else:
        all_ins_u = pd.concat([all_ins_u, region_pm], axis=1)#拼合起来数据
    N = False
    vmlist_u.append(vm)

    if i % 10 == 0:
        print(f"{i} VMs pub_up_bw have been processed")

cols = ["ins_id", "region_id", "ifnull(cpu_rate, 0)", "report_ts"]
cpu = pd.read_csv(root_path+'T_INSTANCE_CPU.csv', usecols=cols)
recpu = cpu[cpu['region_id'].isin(region_list)]
recpu['date'] = pd.to_datetime(recpu['report_ts'], unit='s')
recpu['date'] = pd.DatetimeIndex(recpu['date']) + timedelta(hours=8)
recpu.drop(["report_ts"], axis=1, inplace=True)

region_ins_c = recpu.groupby(['ins_id', 'date'])[['ifnull(cpu_rate, 0)']].mean()

vmlist_c = []
N = True
for i, vm in enumerate(region_ins_c.index.levels[0]):
    region_pm = region_ins_c.loc[vm].resample(args.gra).mean()
    region_pm.rename(columns={'ifnull(cpu_rate, 0)': vm}, inplace=True)
    if len(region_pm) < limits:
        continue
    if np.sum(region_pm.values == 0) >= 1440 or np.sum(region_pm.values == 0) == 720:
        continue
    if N:
        all_ins_c = region_pm
    else:
        all_ins_c = pd.concat([all_ins_c, region_pm], axis=1)
    N = False
    vmlist_c.append(vm)
    if i % 10 == 0:
        print(f"{i} VMs cpu_rate have been processed")

if vmlist_u != vmlist_c:#只保留两列数据都有的vm
    vmlist = [x for x in vmlist_u if x in vmlist_c]
    all_ins_u = all_ins_u[vmlist]
    all_ins_c = all_ins_c[vmlist]
else:
    vmlist = vmlist_u

folder_path = args.output_dir + args.region + '/' + args.gra + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
all_ins_u.to_csv(folder_path + 'up_bw.csv')
all_ins_c.to_csv(folder_path + 'cpu_rate.csv')
print(f"{folder_path}-----{len(vmlist)} VMs have complete data")
vm_d = pd.DataFrame(vmlist, columns=['vm'])
vm_d.to_csv(folder_path + 'vmlist.csv', index=False)
