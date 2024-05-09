#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：lab 
@File ：adjacency_matrix.py
@Author ：Yanan Li
@Date ：2024-03-05 16:25 
@Version :   2.0
@Contact :   YaNanLi@bupt.edu.cn
@Desc    :   节点的三种邻接矩阵
'''
import requests
import json
import math
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def get_location(county):
    '''
    https://lbs.amap.com/api/webservice/guide/api/direction#distance
    :param county: 'changsha'
    :return: '112.938882,28.228304'
    '''
    url = 'https://restapi.amap.com/v3/geocode/geo?parameters'
    params = {'key': '42d27ad6006ed7210e3f59935689cb5e',
              'address': county}

    response = requests.get(url, params=params)
    jd = json.loads(response.content)
    return jd['geocodes'][0]['location']


def get_distance(origin, destination):
    '''

    :param origin:
    :param destination: '112.938882,28.228304'
    :return: '290133'
    '''
    url = 'https://restapi.amap.com/v3/distance?parameters'
    params = {'key': '42d27ad6006ed7210e3f59935689cb5e',
              'origins': origin,
              'destination': destination,
              'type': '0'}  # 参数4：0：直线距离 1：驾车导航距离仅支持国内坐标

    response = requests.get(url, params=params)
    jd = json.loads(response.content)
    return jd['results'][0]['distance']  # unit is m


def correct_adj(matrix, threshold):
    deta = matrix.std()
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ex = math.exp(-1 * pow(matrix[i, j] / deta, 2)) #ex 的计算基于 (i, j) 处元素的值和标准偏差 deta。模拟数值如何以钟形曲线围绕均值分布，其中 x 代表离均值的距离，表达式给出了该距离处的概率密度。
            if matrix[i, j] <= threshold and ex >= 0.1:     #如果元素小于或等于阈值，且 ex 大于或等于 0.1，则更新为 ex
                matrix[i, j] = ex
            else:                                           #
                matrix[i, j] = 0
    return matrix


def global_adj(path):
    dis_threshold=4e5
    rtt_threshold=30
    temp_threshold=40
    df = pd.read_csv(path+'vmlist.csv')
    df = pd.merge(df, df['ens_region_id'].str.split('-', expand=True), left_index=True, right_index=True)
    df.rename(columns={0: 'city', 1: 'ISP', 2: 'num'}, inplace=True)
    site_lst = df['ens_region_id'].tolist()

    # space adjacency
    print('start dis_adj...')
    A_dis = np.zeros((len(site_lst), len(site_lst)), dtype=float)
    diction = {}
    for i in range(len(site_lst)):
        print(f"{100.0*i/len(site_lst):.0f}%\b")
        for j in range(i+1, len(site_lst)):
            
            origin = site_lst[i].split("-")[0]
            dest = site_lst[j].split("-")[0]
            if origin == dest:
                continue
            elif origin+dest in diction:
                A_dis[i, j] = diction[origin+dest]
            elif dest+origin in diction:
                A_dis[i, j] = diction[dest+origin]
            else:
                dis = float(get_distance(get_location(origin), get_location(dest)))
                diction[origin+dest] = dis
                A_dis[i, j] = dis
    B_dis = correct_adj(A_dis + A_dis.T, dis_threshold)
    np.save(path+'remain/dis_adj_'+str(dis_threshold) +
                '.npy', B_dis)

    # temporal adjacency
    print('start tem_adj...')
    '''
    https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw.html#tslearn.metrics.dtw
    https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html#dtw
    '''
    A_tem = np.zeros((len(site_lst), len(site_lst)), dtype=float)
    frame = pd.read_csv(path+"cpu_rate.csv", index_col=0)
    frame = frame[0:int(0.7*len(frame))]  # only computed on the training data
    thres = temp_threshold
    for i in range(len(site_lst)):
        print(f"{100.0*i/len(site_lst):.0f}%\b")
        for j in range(i+1, len(site_lst)):
            origin = frame[site_lst[i]].to_list()
            dest = frame[site_lst[j]].to_list()
            A_tem[i, j] = dtw(origin, dest)  # 越大差异越大，越小越相似
    B_tem = correct_adj(A_tem + A_tem.T, thres)
    np.save(path+'remain/tem_adj_'+str(temp_threshold) +
                '.npy', B_tem)

    # RTT adjacency
    print('start rtt_adj...')
    cn = pd.read_csv(path+'reg2reg_rtt.csv')
    A_rtt = np.zeros((len(site_lst), len(site_lst)), dtype=int)
    for i in range(len(site_lst)):
        print(f"{100.0*i/len(site_lst):.0f}%\b")
        for j in range(i+1, len(site_lst)):
            re1 = cn[cn['from_region_id'] == site_lst[i]]
            re2 = re1[re1['to_region_id'] == site_lst[j]]
            if np.isnan(re2['rtt'].median()):
                A_rtt[i, j] = cn['rtt'].median()
            else:
                A_rtt[i, j] = re2['rtt'].median()
    B_rtt = correct_adj(A_rtt + A_rtt.T, rtt_threshold)
    np.save(path+'remain/rtt_adj_'+str(rtt_threshold) +
                '.npy', B_rtt)
    return B_dis + B_tem + B_rtt


def global_adj_():
    lst = []
    for i in ['dis', 'tem', 'rtt']:
        adj = np.load('/data2/lyn/www23/data/site/15min/' + i + '_adj.npy')
        lst.append(adj)
    return lst



############################################################
def phy_adj(site, freq):
    root_path = '/data2/lyn/www23/data/' + site + '/' + freq + '/'
    df = pd.read_csv(root_path + "vmlist.csv")
    ins = pd.read_csv('/data/ali_ENS/e_vm_instance.csv', usecols=['uuid', 'cores', 'memory', 'storage'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    df1['memory'] = df1['memory'] / 1024
    df1['storage'] = df1['storage'] / 1024
    scl = StandardScaler()
    df_a = scl.fit_transform(df1.values)
    adj = cosine_similarity(df_a)
    return adj


def log_adj(site, freq):
    root_path = '/data2/lyn/www23/data/' + site + '/' + freq + '/'
    df = pd.read_csv(root_path + "vmlist.csv")
    ins = pd.read_csv('/data/ali_ENS/e_vm_instance.csv', usecols=['uuid', 'ali_uid', 'nc_name',
                                                                  'ens_region_id', 'image_id'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    enc = OneHotEncoder()
    df_a = enc.fit_transform(df1.values).toarray()
    adj = cosine_similarity(df_a)
    return adj


def local_adj(site, freq):
    return phy_adj(site, freq) + log_adj(site, freq)


# /data2/lyn/www23/data/site/15min/
# '/data/zph/HGCN/data/site/15min/'
path = 'D:/keyan/2023lyn/HGCN/data/site/15min/'

global_adj(path)