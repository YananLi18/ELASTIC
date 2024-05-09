import pickle
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import requests
import json
import math
from tslearn.metrics import dtw
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List             #informer
from pandas.tseries import offsets  # informer
from pandas.tseries.frequencies import to_offset # informer


class EnsSDatesetRatio(Dataset):
    def __init__(self, root_path='./data/', flag='train', interval=1,
                 data_type='CPU', data='CC', size=[12, 12],
                 scale=False, inverse=False,  freq='15min', cols=None, site='wuhan-telecom'):
        self.interval = interval
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.data_type = data_type
        self.data = data
        self.site = site

        # init
        assert flag in ['train', 'test', 'val']         #检查flag是否是这三个值之一，不是则退出报错
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.inverse = inverse

        self.freq = freq

        self.root_path = root_path
        self.site_path = self.root_path + self.data + '/' + self.freq + '/'
        data_parser = {                                 #数据分析器，决定使用哪个数据
            'CPU': 'cpu_rate.csv',
            'UP': 'up_bw.csv',
            'DOWN': 'down_bw.csv'
        }
        self.data_path = data_parser[data_type]
        self.cols = cols
        self.__read_data__()

    def __read_data__(self):

        vml = pd.read_csv(self.site_path + 'vmlist.csv')
        ins = pd.read_csv(self.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores']) #ins将只包括该表格的这三列（虚拟机id、站点名字、虚拟机核数），不包含其他列
        vmr = pd.merge(vml, ins, how='left', left_on='vm', right_on='uuid')#vml的vm值与ins中uuid值相等的列将合并，执行左合并，vml完全保留，ins对应合并，没有填NAN
        idx = (vmr['ens_region_id'].values == self.site)

        if self.data_type == 'CPU':
            bw_raw = pd.read_csv(self.site_path + self.data_path)
            site_raw = pd.read_csv(self.root_path + 'site' + '/' + self.freq + '/cpu_rate.csv').T
        else:
            bw_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            site_raw = pd.read_csv(self.root_path + 'site' + '/' + self.freq + '/' + self.data_path).T
        bw_raw = bw_raw.fillna(method='ffill', limit=len(bw_raw)).fillna(method='bfill', limit=len(bw_raw))
        bw_raw['date'] = pd.to_datetime(bw_raw['date'])
        bw_raw.set_index('date', inplace=True)

        site_raw.drop('date', inplace=True)
        site_raw.reset_index(inplace=True)
        ss = pd.merge(vmr, site_raw, how='left', left_on='ens_region_id', right_on='index').iloc[:, 5:].T
        ss.columns = bw_raw.columns
        for col in ss.columns.to_list():
            ss[col] = ss[col].map(lambda x: ss[col].median() if x == 0 else x)

        if self.data_type == 'CPU':
            bb = bw_raw.T
            bb.reset_index(inplace=True)
            cpu = pd.merge(bb, ins, how='left', left_on='index', right_on='uuid')
            cpu.iloc[:, 1:1787] = cpu.iloc[:, 1:1787].multiply(cpu.loc[:, 'cores'], axis="index")
            bw_raw = cpu.iloc[:, 1:1787].T

        bw = bw_raw.values / ss.values
        bw = bw[:, idx] * 100
        self.length = len(bw_raw)
        border1s = [0, int(0.7*self.length), int(0.9*self.length)]
        border2s = [int(0.7*self.length), int(0.9*self.length), self.length]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # bw = bw_raw.loc[:, self.cols]
        # bw = bw_raw

        if self.scale:                                  #归一化缩放
            self.scaler = StandardScaler()
            b_u_tr = bw[border1s[0]:border2s[0]]
            self.scaler.fit(b_u_tr.values)
            bu_data = self.scaler.transform(bw.values)
        else:
            bu_data = bw
        self.data_x = bu_data[border1:border2][:, :, np.newaxis]        #它为所选数据添加一个新轴。这样做通常是为了重塑数据

        if self.inverse:                                                #执行可能与预测任务有关的逆运算
            self.data_y = bw.values[border1:border2][:, :, np.newaxis]
        else:
            self.data_y = bu_data[border1:border2][:, :, np.newaxis]    #任务不是反向操作，并且目标数据的准备方式不同。
        
    def __getitem__(self, index):
        lo = index
        hi = lo + self.seq_len
        train_data = self.data_x[lo: hi, :, :]
        target_data = self.data_y[hi:hi + self.pred_len, :, :]
        # x = torch.from_numpy(train_data).type(torch.float)
        # y = torch.from_numpy(target_data).type(torch.float)
        x = torch.from_numpy(train_data).double()
        y = torch.from_numpy(target_data).double()
        return x, y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):#用于反转应用于数据的缩放操作，将其返回到原始缩放比例或单位。
        return self.scaler.inverse_transform(data)
# informer
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth],
        # offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            # DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

def time_features(dates, timeenc=0, freq='t'):
    """
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0: 
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    > 
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]): 
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """

    if timeenc == 0:
        dates['month'] = dates.date.apply(lambda row:row.month,1)
        dates['day'] = dates.date.apply(lambda row:row.day,1)
        dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
        dates['hour'] = dates.date.apply(lambda row:row.hour,1)
        dates['minute'] = dates.date.apply(lambda row:row.minute,1)
        dates['minute'] = dates.minute.map(lambda x:x//15)
        freq_map = {
            'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
            'b':['month','day','weekday'],
            'h':['day','weekday','hour'],# 'h':['month','day','weekday','hour']
            't':['day','weekday','hour','minute'],# 't':[ 'month','day','weekday','hour','minute'],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc == 1:
        dates = pd.to_datetime(dates.date.values)
        see = np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)
    
class EnsSDateset(Dataset):
    def __init__(self, root_path='./data/CC/15min/', flag='train', interval=1,
                 data_type='CPU', data='site', size=[12, 12, 0],
                 scale=False, inverse=False,  freq='15min', site=None,model='gwnet'):
        self.interval = interval
        self.seq_len = size[0]#seq len
        self.pred_len = size[1]#pred len
        self.label_len = size[2]#label len
        self.model = model
        self.data_type = data_type
        self.data = data

        # init
        self.flag =flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.inverse = inverse

        self.freq = freq
        self.root_path = root_path
        self.site_path = self.root_path + self.data + '/' + self.freq + '/'
        data_parser = {
            'CPU': 'cpu_rate.csv',
            'UP': 'up_bw.csv',
            'DOWN': 'down_bw.csv'
        }
        self.data_path = data_parser[data_type]
        self.site = site
        self.__read_data__()

    def __read_data__(self):
        # self.scaler = StandardScaler()
        bw_raw = pd.read_csv(os.path.join(self.site_path, self.data_path))
        bw_raw = bw_raw.fillna(method='ffill', limit=len(bw_raw)).fillna(method='bfill', limit=len(bw_raw))
        bw_raw['date'] = pd.to_datetime(bw_raw['date'])
        bw_raw.set_index('date', inplace=True)
        if self.data != 'site':
            if self.data_type == 'CPU':
                ins = pd.read_csv(self.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores'])
                bb = bw_raw.T              
                bb.reset_index(inplace=True)
                cpu = pd.merge(bb, ins, how='left', left_on='index', right_on='uuid')
                
                cpu.iloc[:, 1:2881] = cpu.iloc[:, 1:2881].multiply(cpu.loc[:, 'cores'], axis="index")
                bw_raw = cpu.iloc[:, 1:2881].T
               
            else:
                bw_raw = bw_raw / 1e6  # bps -> Mbps
        
        if self.freq != '15min':
            bw_raw.index = pd.to_datetime(bw_raw.index)
            bw_raw = bw_raw.resample(self.freq).mean()
        

        if self.site is not None:
            vml = pd.read_csv(self.site_path + 'vmlist.csv')
            ins = pd.read_csv(self.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores'])
            vmr = pd.merge(vml, ins, how='left', left_on='vm', right_on='uuid')
            idx = (vmr['ens_region_id'].values == self.site)                    
            bw_raw = bw_raw.loc[:, idx].copy()                                  
        
        bw_stamp=pd.DataFrame(bw_raw.index, columns=['date'])
        self.length = len(bw_raw)
        
        border1s = [0, int(0.6*self.length), int(0.8*self.length)]              
        border2s = [int(0.6*self.length), int(0.8*self.length), self.length]    

        border1 = border1s[self.set_type]                                       
        border2 = border2s[self.set_type]
        bw = bw_raw.values

        bw_stamp = bw_stamp[border1:border2]# transformer
        bw_stamp = pd.to_datetime(bw_stamp['date']).astype(np.int64)
        self.data_stamp = bw_stamp.values.astype(np.int64)
        if self.scale:
            self.scaler = StandardScaler()
            b_u_tr = bw[border1s[0]:border2s[0]]
            self.scaler.fit(b_u_tr)
            bu_data = self.scaler.transform(bw)
        else:
            bu_data = bw
        self.data_x = bu_data[border1:border2][:, :, np.newaxis]
        if self.inverse:
            self.data_y = bw.values[border1:border2][:, :, np.newaxis]
        else:
            self.data_y = bu_data[border1:border2][:, :, np.newaxis]
        
        if  border2 - border1 < self.seq_len + self.pred_len:
            
            padding_length = self.seq_len + self.pred_len - border2 + border1
            print("padding ", padding_length," pos  for ",self.flag, " dataset")
            
            last_stamp = self.data_stamp[-1] 
            padded_stamps = np.full(padding_length, last_stamp)  
            self.data_stamp = np.concatenate((self.data_stamp, padded_stamps))  

            last_x = self.data_x[-1, :, :]  
            padded_x = np.tile(last_x, (padding_length, 1, 1))  
            self.data_x = np.concatenate((self.data_x, padded_x), axis=0)  

            last_y = self.data_y[-1, :, :]
            padded_y = np.tile(last_y, (padding_length, 1, 1))  
            self.data_y = np.concatenate((self.data_y, padded_y), axis=0) 
        
        
    def __getitem__(self, index):
        lo = index
        hi = lo + self.seq_len

        if self.model=='Informer' or self.model == 'Autoformer' or self.model == 'TimesNet':
            train_data = self.data_x[lo: hi, :, :]
            target_data = self.data_y[hi - self.label_len: hi + self.pred_len, :, :]
        else:
            train_data = self.data_x[lo: hi, :, :]
            target_data = self.data_y[hi:hi + self.pred_len, :, :]
        x = torch.from_numpy(train_data).double()
        y = torch.from_numpy(target_data).double()
        x_mark = self.data_stamp[lo:hi]
        y_mark = self.data_stamp[hi - self.label_len: hi + self.pred_len]
        return x, y, x_mark, y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)


class Dataloader(object):
    def __init__(self, xs, ys, xm, ym, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0            
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            
            xm_date = pd.DataFrame(np.vstack([pd.to_datetime(row, unit='ns') for row in xm]),
                       columns=[f'datetime_col_{i}' for i in range(xm.shape[1])])
            ym_date = pd.DataFrame(np.vstack([pd.to_datetime(row, unit='ns') for row in ym]),
                       columns=[f'datetime_col_{i}' for i in range(ym.shape[1])])
            
            last_xm_values = xm_date.iloc[-1].values
            last_ym_values = ym_date.iloc[-1].values
            padding_interval_seconds = 15 * 60
            padded_xm_values = np.arange(last_xm_values[-1] + padding_interval_seconds,
                                            last_xm_values[-1] + (num_padding + 1) * padding_interval_seconds,
                                            padding_interval_seconds)
            padded_ym_values = np.arange(last_ym_values[-1] + padding_interval_seconds,
                                            last_ym_values[-1] + (num_padding + 1) * padding_interval_seconds,
                                            padding_interval_seconds)
            padded_xm = pd.DataFrame(padded_xm_values.reshape(-1, 1), columns=['datetime_col_0'])
            padded_ym = pd.DataFrame(padded_ym_values.reshape(-1, 1), columns=['datetime_col_0'])
            newx_dates = pd.concat([xm_date, padded_xm], ignore_index=True)    
            newy_dates = pd.concat([ym_date, padded_ym], ignore_index=True)    
            xm = newx_dates
            ym = newy_dates
           
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        

        format_xm = []
        for row in range(xm.shape[0]):
            fuk = [xm.iloc[row,col] for col in range(xm.shape[1])]
            fuk = pd.DataFrame(pd.to_datetime(fuk),columns=['date'])
            fuk = time_features(fuk)
            format_xm.append(fuk)
        self.xm = format_xm
        
        format_ym = []
        for row in range(ym.shape[0]):
            fuk = [ym.iloc[row,col] for col in range(ym.shape[1])]
            fuk = pd.DataFrame(pd.to_datetime(fuk),columns=['date'])
            fuk = time_features(fuk)
            format_ym.append(fuk)
        self.ym = format_ym
        

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        xm, ym = self.xm[permutation], self.ym[permutation]
        self.xs = xs
        self.ys = ys
        self.xm = xm
        self.ym = ym
    def len(self):
        return self.num_batch
    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                xm_i = self.xm[start_ind: end_ind]
                ym_i = self.ym[start_ind: end_ind]
                yield (x_i, y_i, xm_i, ym_i)
                self.current_ind += 1

        return _wrapper()


class DataLoader_cluster(object):#
    def __init__(self, xs, ys,xc,yc, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            num_padding = (batch_size - (len(xc) % batch_size)) % batch_size
            x_padding = np.repeat(xc[-1:], num_padding, axis=0)
            y_padding = np.repeat(yc[-1:], num_padding, axis=0)
            xc = np.concatenate([xc, x_padding], axis=0)
            yc = np.concatenate([yc, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, xc, yc = self.xs[permutation], self.ys[permutation], self.xc[permutation], self.yc[permutation]
        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                x_c = self.xc[start_ind: end_ind, ...]
                y_c = self.yc[start_ind: end_ind, ...]
                yield (x_i, y_i, x_c, y_c)
                self.current_ind += 1

        return _wrapper()


class Standardscaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T]) 
    L = calculate_normalized_laplacian(adj_mx)          
    if lambda_max is None:                             
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')  
        lambda_max = lambda_max[0]                      
    L = sp.csr_matrix(L)                                
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)    
    L = (2 / lambda_max * L) - I                        
    return L.astype(np.float32).todense()               


def load_pickle(pickle_file):                           
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)               
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(root_path, data_type, data_, batch_size,
                 seq_len, pred_len, scaler_flag=False, ratio_flag=False, site=None,lable_len=0,model='gwnet'):
    data = {}
    for category in ['train', 'val', 'test']:
        
        if ratio_flag:
            dataset = EnsSDatesetRatio(root_path=root_path, data_type=data_type, data=data_,
                                       flag=category, size=[seq_len, pred_len], site=site)
        else:
            dataset = EnsSDateset(root_path=root_path, data_type=data_type, data=data_,
                                  flag=category, size=[seq_len, pred_len, lable_len], site=site,model=model)
        dataloader = DataLoader(dataset, batch_size=64)
        for i, (i_x, i_y, i_x_mark, i_y_mark) in enumerate(dataloader):
            if i == 0:
                a_x, a_y, a_x_mark, a_y_mark  = i_x, i_y, i_x_mark, i_y_mark
            a_x = torch.cat((a_x, i_x), dim=0)
            a_y = torch.cat((a_y, i_y), dim=0)
            a_x_mark = torch.cat((a_x_mark, i_x_mark), dim=0)# transformer
            a_y_mark = torch.cat((a_y_mark, i_y_mark), dim=0)# transformer
        data['x_' + category] = a_x.numpy()
        data['y_' + category] = a_y.numpy()
        data['x_mark_' + category] = a_x_mark.numpy()# transformer
        data['y_mark_' + category] = a_y_mark.numpy()# transformer
    # Data format
    if scaler_flag:
        scaler = Standardscaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['scaler'] = scaler
    data['train_loader'] = Dataloader(data['x_train'], data['y_train'], data['x_mark_train'], data['y_mark_train'], batch_size)
    data['val_loader'] = Dataloader(data['x_val'], data['y_val'], data['x_mark_val'], data['y_mark_val'], batch_size)
    data['test_loader'] = Dataloader(data['x_test'], data['y_test'], data['x_mark_test'], data['y_mark_test'], batch_size)

    return data


def load_dataset_cluster(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    for category in ['train_cluster', 'val_cluster', 'test_cluster']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['train_loader_cluster'] = DataLoader_cluster(data['x_train'], data['y_train'],data['x_train_cluster'], data['y_train_cluster'], batch_size)
    data['val_loader_cluster'] = DataLoader_cluster(data['x_val'], data['y_val'],data['x_val_cluster'], data['y_val_cluster'], valid_batch_size)
    data['test_loader_cluster'] = DataLoader_cluster(data['x_test'], data['y_test'],data['x_test_cluster'], data['y_test_cluster'], test_batch_size)
    
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2                   
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def custom_loss_cpu(preds, labels, null_val=np.nan, ratio=1):
    return masked_mse(preds, labels, null_val) * 0.1667 + ratio * masked_mape(preds, labels, null_val)


def custom_loss_cpu_site(preds, labels, null_val=np.nan, ratio=40):
    return masked_mae(preds, labels, null_val) + ratio * masked_mape(preds, labels, null_val)


def custom_loss_bw(preds, labels, null_val=np.nan, ratio=3):
    return masked_mse(preds, labels, null_val)*0.0075 + ratio * masked_mape(preds, labels, null_val)


def custom_loss_bw_site(preds, labels, null_val=np.nan, ratio=500):
    return masked_mae(preds, labels, null_val) + ratio * masked_mape(preds, labels, null_val)

def masked_rmse(preds, labels, null_val=np.nan):
    return masked_mse(preds=preds, labels=labels, null_val=null_val)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)              
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):                      
        mask = ~torch.isnan(labels)             
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)                    
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    eps = 1e-7
    loss = torch.abs(preds - labels) / (torch.abs(labels) + torch.abs(preds) + eps)
    loss = loss * mask                          
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 2                 

def masked_r_squared(preds, labels, null_val=np.nan):
    mean_labels = torch.mean(labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
        mask_labels = ~torch.isnan(mean_labels)
    else:
        mask = (labels != null_val)
        mask_labels = (mean_labels != null_val)
    mask, mask_labels = mask.float(), mask_labels.float()
    mask /= torch.mean(mask)
    mask_labels /= torch.mean(mask_labels)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mask_labels = torch.where(torch.isnan(mask_labels), torch.zeros_like(mask_labels), mask_labels)
    ssr = torch.sum((labels - preds) ** 2 * mask)  
    sst = torch.sum((labels - mean_labels) ** 2 * mask_labels)  
    r_squared = 1.0 - ssr / sst
    return r_squared

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    smape = masked_mape(pred, real, 0.0).item()
    mse = masked_mse(pred, real, 0.0).item()
    r2 = masked_r_squared(pred, real, 0.0).item()
    return mae, smape, mse, r2


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
              'type': '0'}  

    response = requests.get(url, params=params)
    jd = json.loads(response.content)
    return jd['results'][0]['distance']  # unit is m


def correct_adj(matrix, threshold):                         
    deta = matrix.std()                                     
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ex = math.exp(-1 * pow(matrix[i, j] / deta, 2)) 
            if matrix[i, j] <= threshold and ex >= 0.1:     
                matrix[i, j] = ex
            else:                                           
                matrix[i, j] = 0
    return matrix


def global_adj():
    df = pd.read_csv('./data/site/15min/vmlist.csv')
    df = pd.merge(df, df['ens_region_id'].str.split('-', expand=True), left_index=True, right_index=True)
    df.rename(columns={0: 'city', 1: 'ISP', 2: 'num'}, inplace=True)    
    site_lst = df['ens_region_id'].tolist()                            
    # space adjacency
    A_dis = np.zeros((len(site_lst), len(site_lst)), dtype=float)
    diction = {}
    for i in range(len(site_lst)):                                      
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
    B_dis = correct_adj(A_dis + A_dis.T, 4e5)                           
    np.save('./data/site/15min/dis_adj.npy', B_dis)                     

    # temporal adjacency
    '''
    https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw.html#tslearn.metrics.dtw
    https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html#dtw
    '''
    A_tem = np.zeros((len(site_lst), len(site_lst)), dtype=float)
    frame = pd.read_csv("./data/site/15min/cpu_rate.csv", index_col=0)
    frame = frame[0:1786]  # only computed on the training data      
    for i in range(len(site_lst)):
        for j in range(i+1, len(site_lst)):
            origin = frame[site_lst[i]].to_list()                     
            dest = frame[site_lst[j]].to_list()
            A_tem[i, j] = dtw(origin, dest)  
    B_tem = correct_adj(A_tem + A_tem.T, 40)
    np.save('./data/site/15min/tem_adj.npy', B_tem)

    # RTT adjacency
    cn = pd.read_csv('./data/reg2reg_rtt.csv')          
    A_rtt = np.zeros((len(site_lst), len(site_lst)), dtype=float)
    for i in range(len(site_lst)):
        for j in range(i+1, len(site_lst)):
            re1 = cn[cn['from_region_id'] == site_lst[i]]
            re2 = re1[re1['to_region_id'] == site_lst[j]]
            if np.isnan(re2['rtt'].median()):            
                A_rtt[i, j] = cn['rtt'].median()         
            else:                                       
                A_rtt[i, j] = re2['rtt'].median()
    B_rtt = correct_adj(A_rtt + A_rtt.T, 30)
    np.save('./data/site/15min/rtt_adj.npy', B_rtt)
    return B_dis + B_tem + B_rtt


def global_adj_(root_path, dtype, adjtype, dis_threshold=4e5, rtt_threshold=30,
                temp_threshold=40, flag=True):
    lst = []
    df = pd.read_csv(root_path + 'vmlist.csv')
    df = pd.merge(df, df['ens_region_id'].str.split('-', expand=True), left_index=True, right_index=True)
    df.rename(columns={0: 'city', 1: 'ISP', 2: 'num'}, inplace=True)
    site_lst = df['ens_region_id'].tolist()
    if flag:
        # space adjacency
        A_dis = np.zeros((len(site_lst), len(site_lst)), dtype=float)
        tf = open(root_path + "site2site_distance.json", "r")
        diction = json.load(tf)
        save_flag = False
        for i in range(len(site_lst)):
            for j in range(i + 1, len(site_lst)):
                origin = site_lst[i].split("-")[0]
                dest = site_lst[j].split("-")[0]
                if origin == dest:
                    continue
                elif origin + dest in diction:
                    A_dis[i, j] = diction[origin + dest]
                elif dest + origin in diction:
                    A_dis[i, j] = diction[dest + origin]
                else:
                    dis = float(get_distance(get_location(origin), get_location(dest)))
                    diction[origin + dest] = dis
                    A_dis[i, j] = dis
                    save_flag = True
        B_dis = correct_adj(A_dis + A_dis.T, dis_threshold)

        
        A_tem = np.zeros((len(site_lst), len(site_lst)), dtype=float)
        if dtype == 'CPU':
            frame = pd.read_csv(root_path + "cpu_rate.csv", index_col=0)
            frame = frame[0:1786]  # only computed on the training data
            thres = temp_threshold
        else:
            frame = pd.read_csv(root_path + "up_bw.csv", index_col=0)
            frame = frame[0:int(0.7*len(frame))]  # only computed on the training data
            thres = temp_threshold * 1000
        for i in range(len(site_lst)):
            for j in range(i + 1, len(site_lst)):
                origin = frame[site_lst[i]].to_list()
                dest = frame[site_lst[j]].to_list()
                A_tem[i, j] = dtw(origin, dest)  
        B_tem = correct_adj(A_tem + A_tem.T, thres)

        # RTT adjacency
        cn = pd.read_csv(root_path + 'reg2reg_rtt.csv')
        A_rtt = np.zeros((len(site_lst), len(site_lst)), dtype=float)
        for i in range(len(site_lst)):
            for j in range(i + 1, len(site_lst)):
                re1 = cn[cn['from_region_id'] == site_lst[i]]
                re2 = re1[re1['to_region_id'] == site_lst[j]]
                if np.isnan(re2['rtt'].median()):
                    A_rtt[i, j] = cn['rtt'].median()
                else:
                    A_rtt[i, j] = re2['rtt'].median()
        B_rtt = correct_adj(A_rtt + A_rtt.T, rtt_threshold)

        np.save(root_path+'remain/dis_adj_'+str(dis_threshold) +
                '.npy', B_dis)
        np.save(root_path+'remain/tem_adj_'+str(temp_threshold) +
                '.npy', B_tem)
        np.save(root_path+'remain/rtt_adj_'+str(rtt_threshold) +
                '.npy', B_rtt)
        if save_flag:
            tf = open(root_path + "site2site_distance.json", "w")
            json.dump(diction, tf)
            tf.close()
    else:
        B_dis = np.load(root_path+'remain/dis_adj_'+str(dis_threshold)
                        + '.npy')
        B_tem = np.load(root_path+'remain/tem_adj_'+str(temp_threshold)
                        + '.npy')
        B_rtt = np.load(root_path+'remain/rtt_adj_'+str(rtt_threshold)
                        + '.npy')

    if adjtype == 'all1':
        lst.append(B_dis)
        lst.append(B_tem)
        lst.append(B_rtt)
    elif adjtype == 'all2':
        lst.append(B_dis + B_tem + B_rtt)
    elif adjtype == 'dis':
        lst.append(B_dis)
    elif adjtype == 'tem':
        lst.append(B_tem)
    elif adjtype == 'rtt':
        lst.append(B_rtt)
    elif adjtype == 'identity':
        lst.append(np.diag(np.ones(len(site_lst))).astype(np.float32))
    else :
        print("not get a global adj mx.")
    return lst

############################################################


def phy_adj(root_path, site, freq):
    root_path_ = root_path + site + '/' + freq + '/'
    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'cores', 'memory', 'storage'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    df1['memory'] = df1['memory'] / 1024
    df1['storage'] = df1['storage'] / 1024
    scl = OneHotEncoder()
    df_a = scl.fit_transform(df1.values)
    adj = cosine_similarity(df_a)
    return adj


def log_adj(root_path, site, freq):
    root_path_ = root_path + site + '/' + freq + '/'
    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'ali_uid', 'nc_name',
                      'ens_region_id', 'image_id'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    enc = OneHotEncoder()
    df_a = enc.fit_transform(df1.values).toarray()
    adj = cosine_similarity(df_a)
    return adj


def local_adj_(root_path, site, freq, adjtype):
    lst = []
    if adjtype == 'all':
        lst.append(phy_adj(root_path, site, freq))
        lst.append(log_adj(root_path, site, freq))
    elif adjtype == 'phy':
        lst.append(phy_adj(root_path, site, freq))
    elif adjtype == 'log':
        lst.append(log_adj(root_path, site, freq))
    elif adjtype == 'identity':
        adj_mx = log_adj(root_path, site, freq)
        lst.append(np.diag(np.ones(adj_mx.shape[0])).astype(np.float32))
    else :
        print("not get a local adj mx.")
    return lst


def site_index(root_path, data, freq, site):
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id'])
    root_path_ = root_path + data + '/' + freq + '/'
    vmlist = pd.read_csv(root_path_ + 'vmlist.csv')

    need_vm = ins[ins['ens_region_id'] == site]['uuid']
    is_in_vmlist=vmlist['vm'].isin(need_vm)
    indices = np.where(is_in_vmlist)[0]
    indices_list = indices.tolist()
    return indices_list



def RNC_loss(features, labels, tau=1.25):
        similarities = -torch.cdist(features, features, p=2) / tau 

        batch_size = features.size(0) 

        labels_exp = labels.view(-1, 1)
        label_differences = torch.abs(labels_exp - labels_exp.t())

        mask_diagonal = torch.eye(batch_size, dtype=torch.bool).to(features.device)
        greater_mask = label_differences.t() > label_differences

        similarities_exp = torch.exp(similarities)
        similarities_exp = similarities_exp.masked_fill(mask_diagonal, 0)
        denominator = torch.sum(similarities_exp * greater_mask, dim=1, keepdim=True)

        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        p_ij = similarities_exp / denominator

        loss = -torch.sum(torch.log(p_ij + 1e-9)) / (batch_size * (batch_size - 1))
        return loss
def NT_Xent_loss(features, labels, tau=1.25):
    
    similarities = torch.mm(features, features.t()) / tau 

    batch_size = features.size(0)

    mask_diagonal = torch.eye(batch_size, dtype=torch.bool).to(features.device)

    exp_similarities = torch.exp(similarities)
    exp_similarities = exp_similarities.masked_fill(mask_diagonal, 0)

   
    labels_match = torch.eq(labels, labels.t()).float()

    sum_exp_similarities = torch.sum(exp_similarities, dim=1, keepdim=True)

    loss = -torch.sum(torch.log(exp_similarities * labels_match / sum_exp_similarities + 1e-9)) / batch_size
    return loss
def NT_Logistic_loss(features, labels, tau=1.25):
    
    similarities = torch.mm(features, features.t()) / tau 

    batch_size = features.size(0)

    mask_diagonal = torch.eye(batch_size, dtype=torch.bool).to(features.device)

    sigmoid_similarities = torch.sigmoid(similarities)
    sigmoid_similarities = sigmoid_similarities.masked_fill(mask_diagonal, 0)

    labels_match = torch.eq(labels, labels.t()).float()


    loss = -torch.sum(labels_match * torch.log(sigmoid_similarities + 1e-9) +
                      (1 - labels_match) * torch.log(1 - sigmoid_similarities + 1e-9)) / batch_size
    return loss

def Margin_Triplet_loss(features, labels, margin=1.25):

    distance_matrix = torch.cdist(features, features, p=2)  

    batch_size = features.size(0)

    labels_eq = torch.eq(labels, labels.t()).float()

    positive_distances = distance_matrix * labels_eq

    negative_distances = distance_matrix * (1 - labels_eq)

    loss = torch.max(positive_distances - negative_distances + margin, torch.zeros_like(distance_matrix)).mean()
    return loss

