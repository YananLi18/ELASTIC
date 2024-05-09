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
        bw_raw = pd.read_csv(os.path.join(self.site_path, self.data_path))#合并变量形成具体路径，前者为路径名，后者为csv文件名
        bw_raw = bw_raw.fillna(method='ffill', limit=len(bw_raw)).fillna(method='bfill', limit=len(bw_raw))#使用前向填充和后向填充两种方法尽可能多地填充数据帧 bw_raw 中的缺失值。用附近的有效值替换 DataFrame 中的 NaN 值。
        bw_raw['date'] = pd.to_datetime(bw_raw['date'])#使用 pandas 库中的 pd.to_datetime 函数将数据帧 bw_raw 中名为 "date "的列转换为日期格式
        bw_raw.set_index('date', inplace=True)#使用了 pandas 库中的 set_index 方法，将 "data "列设置为 bw_raw 的索引。就地改变结构，使 "日期 "成为新的索引，这在处理时间序列数据时很常见。
        if self.data != 'site':
            if self.data_type == 'CPU':
                ins = pd.read_csv(self.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores'])
                bb = bw_raw.T               #转置会导致索引行失效，所以下一步要处理索引，转置是为了让uid变成csv的一列，才能和ins合并
                bb.reset_index(inplace=True)# 重置索引时，当前索引将移回 DataFrame 中的常规列，并分配一个新的index索引
                cpu = pd.merge(bb, ins, how='left', left_on='index', right_on='uuid')#这里的index在上一行处理后，就是uuid
                
                cpu.iloc[:, 1:2881] = cpu.iloc[:, 1:2881].multiply(cpu.loc[:, 'cores'], axis="index")
                bw_raw = cpu.iloc[:, 1:2881].T
                #cpu.iloc[:, 1:1787] = cpu.iloc[:, 1:1787].multiply(cpu.loc[:, 'cores'], axis="index")
                #第1列到第1786列中的每个值都与 "cores "列中的相应值相乘，矩阵尺寸不变，这么做的指示器就是axis=”index“，保证了index值相同的两个数相乘
                #bw_raw = cpu.iloc[:, 1:1787].T#再转置回去，只保留了cpu_rate的数据，index是时间，bw_raw[i]是一个时间序列
            else:
                bw_raw = bw_raw / 1e6  # bps -> Mbps
        # self.freq = '1H'
        if self.freq != '15min':
            bw_raw.index = pd.to_datetime(bw_raw.index)
            bw_raw = bw_raw.resample(self.freq).mean()#按照数据重新采样到一个较低的频率，然后取指定频率内每组数据点的平均值。对于新频率中的每个时间段，您将获得该时间段内数据点的平均值
        print("freq:",self.freq,"len:",len(bw_raw))

        if self.site is not None:
            vml = pd.read_csv(self.site_path + 'vmlist.csv')
            ins = pd.read_csv(self.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores'])
            vmr = pd.merge(vml, ins, how='left', left_on='vm', right_on='uuid')
            idx = (vmr['ens_region_id'].values == self.site)                    #创建一个bool列，只包括站点名为site值的列
            bw_raw = bw_raw.loc[:, idx].copy()                                  #通过idx枚举bw_raw的每一列，选取bool值为1的一列数据
        # print(bw_raw) 203列,index是时间序列
        # informer <class 'pandas.core.frame.DataFrame'>
        bw_stamp=pd.DataFrame(bw_raw.index, columns=['date'])
        self.length = len(bw_raw)
        #print(bw_raw)
        border1s = [0, int(0.6*self.length), int(0.8*self.length)]              #分割训练、测试、评估集的大小，7：2：1。border1表示左端点
        border2s = [int(0.6*self.length), int(0.8*self.length), self.length]    #border2表示区间右端点

        border1 = border1s[self.set_type]                                       #根据当前操作类型选择区间位置
        border2 = border2s[self.set_type]
        # bw = bw_raw.loc[:, self.cols]
        bw = bw_raw.values

        bw_stamp = bw_stamp[border1:border2]# informer
        bw_stamp = pd.to_datetime(bw_stamp['date']).astype(np.int64)
        self.data_stamp = bw_stamp.values.astype(np.int64)# informer 还有一次枚举，所以这里不做feature提取，但是DataFrame不能枚举，所以要转换一下
        if self.scale:
            self.scaler = StandardScaler()
            b_u_tr = bw[border1s[0]:border2s[0]]
            self.scaler.fit(b_u_tr)#StandardScaler计算b_u_tr中每个特征（列）的平均值和标准偏差
            bu_data = self.scaler.transform(bw)#根据b_u_tr计算的平均值和标准偏差值，以bw.value转换（缩放）数据。转换后的数据现在已标准化，存储在变量bu_data中。
        else:
            bu_data = bw
        self.data_x = bu_data[border1:border2][:, :, np.newaxis]#增加了第三个维度，使得数据变成了（a*b*1）大小，第三维大小为1
        if self.inverse:
            self.data_y = bw.values[border1:border2][:, :, np.newaxis]
        else:
            self.data_y = bu_data[border1:border2][:, :, np.newaxis]
        
        if  border2 - border1 < self.seq_len + self.pred_len:
            
            padding_length = self.seq_len + self.pred_len - border2 + border1
            print("padding ", padding_length," pos  for ",self.flag, " dataset")
            # 为时间戳填充
            last_stamp = self.data_stamp[-1]  # 获取最后一个时间戳
            padded_stamps = np.full(padding_length, last_stamp)  # 创建重复的时间戳数组进行填充
            self.data_stamp = np.concatenate((self.data_stamp, padded_stamps))  # 合并原始数据和填充数据

            # 为 data_x 填充
            last_x = self.data_x[-1, :, :]  # 获取最后一条数据
            padded_x = np.tile(last_x, (padding_length, 1, 1))  # 创建重复的数据数组进行填充
            self.data_x = np.concatenate((self.data_x, padded_x), axis=0)  # 合并原始数据和填充数据

            # 为 data_y 填充
            last_y = self.data_y[-1, :, :]
            padded_y = np.tile(last_y, (padding_length, 1, 1))  # 创建重复的数据数组进行填充
            self.data_y = np.concatenate((self.data_y, padded_y), axis=0)  # 合并原始数据和填充数据
        
        
    def __getitem__(self, index):
        lo = index
        hi = lo + self.seq_len

        if self.model=='Informer' or self.model == 'Autoformer' or self.model == 'TimesNet':
            train_data = self.data_x[lo: hi, :, :]
            target_data = self.data_y[hi - self.label_len: hi + self.pred_len, :, :]# y 序列长为lable+pred长度
        else:
            train_data = self.data_x[lo: hi, :, :]
            target_data = self.data_y[hi:hi + self.pred_len, :, :]
        # x = torch.from_numpy(train_data).type(torch.float)
        # y = torch.from_numpy(target_data).type(torch.float)
        x = torch.from_numpy(train_data).double()
        y = torch.from_numpy(target_data).double()
        x_mark = self.data_stamp[lo:hi]# informer
        y_mark = self.data_stamp[hi - self.label_len: hi + self.pred_len]# informer
        return x, y, x_mark, y_mark# informer

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)


class Dataloader(object):
    def __init__(self, xs, ys, xm, ym, batch_size, pad_with_last_sample=True):
        """
        :param xs:输入数据示例
        :param ys:对应的目标值或标签
        :param batch_size:小批量训练所需的批量大小
        :param pad_with_last_sample: 如果样本数不能被`batch_size`整除，则确定是否用最后一个样本填充数据的标志。默认情况下，它设置为“True” pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0            #用于在批处理迭代期间跟踪当前索引。
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size#计算填充数据所需的样本数，使其可被`batch_ssize`整除
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)#通过分别重复“xs”的最后一个样本“num_padding”次而创建的。此填充可确保最后一个批次的大小与其他批次的大小相同。
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)#将“x_padding”连接到原始数据来扩展“xs”和“ys”
            ys = np.concatenate([ys, y_padding], axis=0)
            
            xm_date = pd.DataFrame(np.vstack([pd.to_datetime(row, unit='ns') for row in xm]),
                       columns=[f'datetime_col_{i}' for i in range(xm.shape[1])])#[1291,12]
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
            # xm = np.concatenate([xm, newx_dates])        
            # ym = np.concatenate([ym, newy_dates])
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs#[1344,12,66,1]
        self.ys = ys
        

        format_xm = []
        for row in range(xm.shape[0]):
            fuk = [xm.iloc[row,col] for col in range(xm.shape[1])]#枚举一列[12,1]
            fuk = pd.DataFrame(pd.to_datetime(fuk),columns=['date'])#转换为名为date的日期
            fuk = time_features(fuk)#处理为[12,6],6分别为date,month,day,weekday,hour,min
            format_xm.append(fuk)
        self.xm = format_xm
        
        format_ym = []
        for row in range(ym.shape[0]):
            fuk = [ym.iloc[row,col] for col in range(ym.shape[1])]#枚举一列[12,1]
            fuk = pd.DataFrame(pd.to_datetime(fuk),columns=['date'])#转换为名为date的日期
            fuk = time_features(fuk)#处理为[12,6],6分别为date,month,day,weekday,hour,min
            format_ym.append(fuk)
        self.ym = format_ym
        

    def shuffle(self):#用于在数据集中打乱数据样本及其相应标签的顺序，引入了随机性，并阻止模型学习数据中的任何顺序或序列。
        permutation = np.random.permutation(self.size)#这行生成从0到`self.size-1`的整数的随机置换。该排列表示数据样本和标签将被排列的新顺序
        xs, ys = self.xs[permutation], self.ys[permutation]#使用上一步中生成的排列，指定的新顺序重新排列`xs'中的数据样本及其在`ys'中的相应标签。这有效地搅乱了数据集。
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
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
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


def sym_adj(adj):#对称归一化邻接矩阵，以确保邻接矩阵可以有效地用于图卷积。
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)#这行使用sp.coo_matrix函数将输入邻接矩阵adj转换为coo（坐标列表）格式的稀疏矩阵。COO格式是表示稀疏矩阵的一种有效的内存方式。
    rowsum = np.array(adj.sum(1))#计算邻接矩阵中每行的和，并将结果存储在rowsum变量中
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()#计算行和的平方根倒数，并对所得数组进行平坦化。
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.#如果行和为零或非常接近零，则结果将为inf，任何inf元素（由于零行和）设置为0。这一步骤对于数值稳定性非常重要
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)#使用d_inv_sqt中的值构造稀疏对角矩阵，将值按对角线填入对角矩阵中
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()#进行对称归一化。


def asym_adj(adj):#非对称归一化
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()#latten（）方法用于确保rowsum是一个一维数组。
    d_inv = np.power(rowsum, -1).flatten()#计算行和的倒数，并对所得数组进行平坦化。逆运算按元素应用于行和数组。
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):#从邻接矩阵计算归一化图的拉普拉斯矩阵，表示图的归一化拉普拉斯算子。它是谱图理论中的一个关键矩阵，用于各种基于图的机器学习算法，用于聚类和降维等任务s
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
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()#使用公式L=I-d^（-0.5）*A*d^（-0.5）计算归一化图拉普拉斯矩阵
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):#计算图的缩放拉普拉斯矩阵。缩放参数lambda_max，用于确定拉普拉斯矩阵的缩放。
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])  #通过取`adj_mx`的元素最大值及其转置来确保输入邻接矩阵是对称的（无向的），确保即使原始数据包含不对称边，图也被视为无向图
    L = calculate_normalized_laplacian(adj_mx)          #获得缩放拉普拉斯算子
    if lambda_max is None:                              #计算归一化拉普拉斯矩阵的最大特征值（谱半径）
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')  #使用Lanczos算法和`LM'（最大幅值）选项计算归一化拉普拉斯矩阵`L'的最大特征值
        lambda_max = lambda_max[0]                      #提取最大的特征值
    L = sp.csr_matrix(L)                                #转换为压缩稀疏行（csr）矩阵格式。CSR格式是一种用于数值计算的高效稀疏矩阵格式
    M, _ = L.shape#拉普拉斯矩阵中的行数
    I = sp.identity(M, format='csr', dtype=L.dtype)     #构造了与拉普拉斯矩阵具有相同行数的单位矩阵（“I”）。矩阵以CSR格式表示。
    L = (2 / lambda_max * L) - I                        #缩放拉普拉斯矩阵“L”。它从按“2/lambda_max”缩放的归一化拉普拉斯矩阵“L”中减去单位矩阵“I”
    return L.astype(np.float32).todense()               #转换为数据类型为“float32”的稠密矩阵并返回


def load_pickle(pickle_file):                           #Pickle是一种Python序列化格式，允许您存储和检索Python对象
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)                #pickle.loard函数用于反序列化和加载打开的pickle文件中的数据。此行读取序列化的数据并将其存储在pickle_data变量中。
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
                 seq_len, pred_len, scaler_flag=False, ratio_flag=False, site=None,lable_len=0,model='gwnet'):#加载时间序列数据，创建数据加载器，并在需要时应用数据缩放
    data = {}
    for category in ['train', 'val', 'test']:
        # cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        if ratio_flag:
            dataset = EnsSDatesetRatio(root_path=root_path, data_type=data_type, data=data_,
                                       flag=category, size=[seq_len, pred_len], site=site)
        else:
            dataset = EnsSDateset(root_path=root_path, data_type=data_type, data=data_,
                                  flag=category, size=[seq_len, pred_len, lable_len], site=site,model=model)
        dataloader = DataLoader(dataset, batch_size=64)
        for i, (i_x, i_y, i_x_mark, i_y_mark) in enumerate(dataloader):#它使用数据加载器来批量加载数据，其中“i_x”表示输入序列，“i_y”表示目标序列
            if i == 0:#输入序列（“i_x”）和目标序列（“i_y”）沿着批维度连接起来，有效地将给定类别的所有批次合并为一个批次
                a_x, a_y, a_x_mark, a_y_mark  = i_x, i_y, i_x_mark, i_y_mark
            a_x = torch.cat((a_x, i_x), dim=0)
            a_y = torch.cat((a_y, i_y), dim=0)
            a_x_mark = torch.cat((a_x_mark, i_x_mark), dim=0)# informer
            a_y_mark = torch.cat((a_y_mark, i_y_mark), dim=0)# informer
        data['x_' + category] = a_x.numpy()#输入序列和目标序列存储在“数据”字典中
        data['y_' + category] = a_y.numpy()
        data['x_mark_' + category] = a_x_mark.numpy()# informer
        data['y_mark_' + category] = a_y_mark.numpy()# informer
    # Data format
    if scaler_flag:
        scaler = Standardscaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())#它计算“x_train”数据的平均值和标准偏差
        for category in ['train', 'val', 'test']:#，并对“x_train”、“x_val”和“x_test”应用相同的缩放比例。
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])#缩放不会改变数据集大小
        data['scaler'] = scaler
    data['train_loader'] = Dataloader(data['x_train'], data['y_train'], data['x_mark_train'], data['y_mark_train'], batch_size)#数据加载程序分别存储在“data”字典中
    data['val_loader'] = Dataloader(data['x_val'], data['y_val'], data['x_mark_val'], data['y_mark_val'], batch_size)
    data['test_loader'] = Dataloader(data['x_test'], data['y_test'], data['x_mark_test'], data['y_mark_test'], batch_size)

    return data


def load_dataset_cluster(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))#连接两者来构造npz文件的路径
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


def masked_mse(preds, labels, null_val=np.nan): #屏蔽均方误差（mse）
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2                    #平方误差（平方差）
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


# def masked_rmse(preds, labels, null_val=np.nan):
#     return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
def custom_loss_cpu(preds, labels, null_val=np.nan, ratio=1):#损失组合了两个不同的损失分量：均方误差（MSE）和平均绝对百分比误差（MAPE）
    return masked_mse(preds, labels, null_val) * 0.1667 + ratio * masked_mape(preds, labels, null_val)#控制MAPE分量相对于MSE的权重的参数


def custom_loss_cpu_site(preds, labels, null_val=np.nan, ratio=40):
    return masked_mae(preds, labels, null_val) + ratio * masked_mape(preds, labels, null_val)


def custom_loss_bw(preds, labels, null_val=np.nan, ratio=3):
    return masked_mse(preds, labels, null_val)*0.0075 + ratio * masked_mape(preds, labels, null_val)


def custom_loss_bw_site(preds, labels, null_val=np.nan, ratio=500):
    return masked_mae(preds, labels, null_val) + ratio * masked_mape(preds, labels, null_val)

def masked_rmse(preds, labels, null_val=np.nan):
    return masked_mse(preds=preds, labels=labels, null_val=null_val)


def masked_mae(preds, labels, null_val=np.nan):#屏蔽平均绝对误差（mae）
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)              #计算preds和标签之间的绝对误差（绝对差）。
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):#掩蔽平均绝对百分比误差
    if np.isnan(null_val):                      #代码定义了一个名为mask的布尔掩码，用于确定标签中的哪些元素应包含在计算中
        mask = ~torch.isnan(labels)             #其中True对应于labels数组中的非NaN值
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)                    #通过将掩码除以其自身的平均值来对掩码进行归一化。此步骤可确保遮罩的平均值为1
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)#此行检查掩码中的任何NaN值，并将其替换为零，以确保NaN值不会影响计算。
    # loss = torch.abs(preds-labels)/torch.add(labels, preds)#将计算每对相应的pred和标签的绝对百分比误差。informer的pred有负数
    eps = 1e-7
    loss = torch.abs(preds - labels) / (torch.abs(labels) + torch.abs(preds) + eps)
    loss = loss * mask                          #损失值按元素乘以掩码，以排除掩码为零（或NaN）的值
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)#损失张量中的任何剩余NaN值都被替换为零。
    return torch.mean(loss) * 2                 #计算屏蔽损失值的平均值，并将其乘以2，然后将其作为最终结果返回。

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
    ssr = torch.sum((labels - preds) ** 2 * mask)  # 残差平方和
    sst = torch.sum((labels - mean_labels) ** 2 * mask_labels)  # 总平方和
    r_squared = 1.0 - ssr / sst
    return r_squared

def metric(pred, real):#用于预测阶段，评估预测数据和真实值之间的量化差距
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
              'type': '0'}  # 参数4：0：直线距离 1：驾车导航距离仅支持国内坐标

    response = requests.get(url, params=params)
    jd = json.loads(response.content)
    return jd['results'][0]['distance']  # unit is m


def correct_adj(matrix, threshold):                         #阈值为4e5
    deta = matrix.std()                                     #deta 计算的是整个矩阵的标准偏差
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ex = math.exp(-1 * pow(matrix[i, j] / deta, 2)) #ex 的计算基于 (i, j) 处元素的值和标准偏差 deta。模拟数值如何以钟形曲线围绕均值分布，其中 x 代表离均值的距离，表达式给出了该距离处的概率密度。
            if matrix[i, j] <= threshold and ex >= 0.1:     #如果元素小于或等于阈值，且 ex 大于或等于 0.1，则更新为 ex
                matrix[i, j] = ex
            else:                                           #
                matrix[i, j] = 0
    return matrix


def global_adj():
    df = pd.read_csv('./data/site/15min/vmlist.csv')
    df = pd.merge(df, df['ens_region_id'].str.split('-', expand=True), left_index=True, right_index=True)
    df.rename(columns={0: 'city', 1: 'ISP', 2: 'num'}, inplace=True)    #按照三个部分把vm站点的名字分为三个部分
    site_lst = df['ens_region_id'].tolist()                             #站点全称的一行形式的张量
    # space adjacency
    A_dis = np.zeros((len(site_lst), len(site_lst)), dtype=float)
    diction = {}
    for i in range(len(site_lst)):                                      #枚举任意两个站点i，j，使得ij不重复
        for j in range(i+1, len(site_lst)):
            origin = site_lst[i].split("-")[0]                          #只截取城市名字
            dest = site_lst[j].split("-")[0]
            if origin == dest:
                continue
            elif origin+dest in diction:                                #两个名字拼音连接作为字典标注，表示是否写入了两个城市间的距离
                A_dis[i, j] = diction[origin+dest]
            elif dest+origin in diction:
                A_dis[i, j] = diction[dest+origin]
            else:
                dis = float(get_distance(get_location(origin), get_location(dest)))
                diction[origin+dest] = dis
                A_dis[i, j] = dis
    B_dis = correct_adj(A_dis + A_dis.T, 4e5)                           #因为ij的减半枚举方式，导致A_dis张量的下半没有写入，于自己的转置相加相当于对称的填好邻接矩阵
    np.save('./data/site/15min/dis_adj.npy', B_dis)                     #两地间距离的概率密度矩阵，舍弃极近距离

    # temporal adjacency
    '''
    https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw.html#tslearn.metrics.dtw
    https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html#dtw
    '''
    A_tem = np.zeros((len(site_lst), len(site_lst)), dtype=float)
    frame = pd.read_csv("./data/site/15min/cpu_rate.csv", index_col=0)#表格里第一行是每列的名称，py会当作字典索引，而不是第一行
    frame = frame[0:1786]  # only computed on the training data       #所以[0:1786]就是从第二行到最后一行的数据
    for i in range(len(site_lst)):
        for j in range(i+1, len(site_lst)):
            origin = frame[site_lst[i]].to_list()                     #枚举每个城市的时间序列，就是csv表格的一列
            dest = frame[site_lst[j]].to_list()
            A_tem[i, j] = dtw(origin, dest)  # 越大差异越大，越小越相似
    B_tem = correct_adj(A_tem + A_tem.T, 40)
    np.save('./data/site/15min/tem_adj.npy', B_tem)

    # RTT adjacency
    cn = pd.read_csv('./data/reg2reg_rtt.csv')           #站点间的往返时间RTT表格，rtt表示了站点间的网络传输距离
    A_rtt = np.zeros((len(site_lst), len(site_lst)), dtype=float)
    for i in range(len(site_lst)):
        for j in range(i+1, len(site_lst)):
            re1 = cn[cn['from_region_id'] == site_lst[i]]#选择from站点为i站点的A_rtt中的所有行
            re2 = re1[re1['to_region_id'] == site_lst[j]]#选择to站点为j站点的re1中的所有行
            if np.isnan(re2['rtt'].median()):            #如果这些值的中值为NAN，可能cn中并不存在这个数据
                A_rtt[i, j] = cn['rtt'].median()         #就处理为整个数组的中值
            else:                                        #如果是数字，就将ij站点间的rtt设置为数据交互的中值
                A_rtt[i, j] = re2['rtt'].median()
    B_rtt = correct_adj(A_rtt + A_rtt.T, 30)
    np.save('./data/site/15min/rtt_adj.npy', B_rtt)
    return B_dis + B_tem + B_rtt


def global_adj_(root_path, dtype, adjtype, dis_threshold=4e5, rtt_threshold=30,
                temp_threshold=40, flag=True):
    print("global adj mx type:", adjtype)
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

        # temporal adjacency
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
                A_tem[i, j] = dtw(origin, dest)  # 越大差异越大，越小越相似
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


def phy_adj(root_path, site, freq):#adj 相似性矩阵，每个vm物理配置配置的相似性
    root_path_ = root_path + site + '/' + freq + '/'
    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'cores', 'memory', 'storage'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)#指定要删除“vm”和“uuid”列，DataFrame将在删除这些列的情况下进行更新。一行表示一个vm
    df1['memory'] = df1['memory'] / 1024
    df1['storage'] = df1['storage'] / 1024
    # scl = StandardScaler()
    scl = OneHotEncoder()#执行了一次热编码，
    df_a = scl.fit_transform(df1.values)#执行单热编码，并返回包含数据的单热编码表示的稀疏矩阵
    adj = cosine_similarity(df_a)#计算该矩阵中所有行对之间的余弦相似度。得到相似性矩阵（adj），其中每个条目表示一个热编码数据中两行之间的余弦相似性，即每对VM之间的相似性。
    return adj


def log_adj(root_path, site, freq):#每个vm的逻辑配置相似性，即网络连接拓扑关系的相似性
    root_path_ = root_path + site + '/' + freq + '/'
    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'ali_uid', 'nc_name',
                      'ens_region_id', 'image_id'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    enc = OneHotEncoder()
    df_a = enc.fit_transform(df1.values).toarray()#在应用编码后，结果通常是表示编码数据的稀疏矩阵。“toarray（）”方法用于将此稀疏矩阵转换为密集NumPy数组
    adj = cosine_similarity(df_a)
    return adj


def local_adj_(root_path, site, freq, adjtype):
    print("local adj mx type:", adjtype)
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
        lst.append(np.diag(np.ones(adj_mx.shape[0])).astype(np.float32))#创建了一个在主对角线上有1，在其他地方有0的方阵，大小为邻接矩阵中的行（或节点）数，邻接矩阵中的行（或节点）数
    else :
        print("not get a local adj mx.")
    return lst


def site_index(root_path, data, freq, site):
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id'])
    root_path_ = root_path + data + '/' + freq + '/'
    vmlist = pd.read_csv(root_path_ + 'vmlist.csv')
    # mx = 0
    # ans = 'none'
    # tot = 0
    # sitelist = pd.read_csv('/data/zph/HGCN/data/site/15min/' + 'vmlist.csv')
    # for id in sitelist['ens_region_id']:
    #     allvminsite = ins[ins['ens_region_id'] == id]['uuid']
    #     needvminsite = vmlist['vm'].isin(allvminsite)
    #     sum = np.where(needvminsite)[0].tolist()
    #     print(id,len(sum))
    #     tot += len(sum)
    #     if len(sum) > mx:
    #         mx = len(sum)
    #         ans = id
    # print(mx,ans,tot)
    need_vm = ins[ins['ens_region_id'] == site]['uuid']
    is_in_vmlist=vmlist['vm'].isin(need_vm)
    indices = np.where(is_in_vmlist)[0]
    indices_list = indices.tolist()
    return indices_list



def RNC_loss(features, labels, tau=1.25):
        """
        Compute the per-sample Rank-N-Contrast (RNC) loss.

        :param features: Tensor of shape (batch_size, 1, features, len), representing the feature embeddings.
        :param labels: Tensor of shape (batch_size, 1, 1, len), representing the target labels.
        :param tau: Temperature parameter.
        :return: The per-sample RNC loss.
        """
        # similarities = torch.mm(features, features.t()) / tau  # 点积计算方式
        similarities = -torch.cdist(features, features, p=2) / tau # 向量之间L2距离的负值

        batch_size = features.size(0) # batch_size = batch*len

        # Expand labels to compute the pairwise label differences
        labels_exp = labels.view(-1, 1)
        label_differences = torch.abs(labels_exp - labels_exp.t())

        # Mask for excluding the diagonal (self-similarity) and get the ranking mask
        mask_diagonal = torch.eye(batch_size, dtype=torch.bool).to(features.device)
        greater_mask = label_differences.t() > label_differences

        # Apply masks and compute the per-sample RNC loss
        similarities_exp = torch.exp(similarities)
        similarities_exp = similarities_exp.masked_fill(mask_diagonal, 0)
        denominator = torch.sum(similarities_exp * greater_mask, dim=1, keepdim=True)

        # Avoid division by zero
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        p_ij = similarities_exp / denominator

        # Taking the log and averaging over all non-diagonal entries
        loss = -torch.sum(torch.log(p_ij + 1e-9)) / (batch_size * (batch_size - 1))
        return loss
def NT_Xent_loss(features, labels, tau=1.25):
    """
    Compute the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    :param features: Tensor of shape (batch_size, 1, features, len), representing the feature embeddings.
    :param labels: Tensor of shape (batch_size, 1, 1, len), representing the target labels.
    :param tau: Temperature parameter.
    :return: The per-sample NT-Xent loss.
    """
    similarities = torch.mm(features, features.t()) / tau  # Dot product similarity

    batch_size = features.size(0)

    # Mask for excluding the diagonal (self-similarity)
    mask_diagonal = torch.eye(batch_size, dtype=torch.bool).to(features.device)

    # Compute the exponential similarities and set diagonal elements to zero
    exp_similarities = torch.exp(similarities)
    exp_similarities = exp_similarities.masked_fill(mask_diagonal, 0)

    # Labels match matrix (1 for matching labels, 0 otherwise)
    labels_match = torch.eq(labels, labels.t()).float()

    # Sum of exp similarities for normalization
    sum_exp_similarities = torch.sum(exp_similarities, dim=1, keepdim=True)

    # Loss calculation
    loss = -torch.sum(torch.log(exp_similarities * labels_match / sum_exp_similarities + 1e-9)) / batch_size
    return loss
def NT_Logistic_loss(features, labels, tau=1.25):
    """
    Compute the NT-Logistic loss.

    :param features: Tensor of shape (batch_size, 1, features, len), representing the feature embeddings.
    :param labels: Tensor of shape (batch_size, 1, 1, len), representing the target labels.
    :param tau: Temperature parameter.
    :return: The per-sample NT-Logistic loss.
    """
    similarities = torch.mm(features, features.t()) / tau  # Dot product similarity

    batch_size = features.size(0)

    # Mask for excluding the diagonal (self-similarity)
    mask_diagonal = torch.eye(batch_size, dtype=torch.bool).to(features.device)

    # Compute the sigmoid of similarities and set diagonal elements to zero
    sigmoid_similarities = torch.sigmoid(similarities)
    sigmoid_similarities = sigmoid_similarities.masked_fill(mask_diagonal, 0)

    # Labels match matrix (1 for matching labels, 0 otherwise)
    labels_match = torch.eq(labels, labels.t()).float()

    # Binary cross-entropy loss
    loss = -torch.sum(labels_match * torch.log(sigmoid_similarities + 1e-9) +
                      (1 - labels_match) * torch.log(1 - sigmoid_similarities + 1e-9)) / batch_size
    return loss

def Margin_Triplet_loss(features, labels, margin=1.25):
    """
    Compute the Margin Triplet loss.

    :param features: Tensor of shape (batch_size, 1, features, len), representing the feature embeddings.
    :param labels: Tensor of shape (batch_size, 1, 1, len), representing the target labels.
    :param margin: Margin parameter.
    :return: The per-sample Margin Triplet loss.
    """
    distance_matrix = torch.cdist(features, features, p=2)  # Pairwise Euclidean distances

    batch_size = features.size(0)

    # Create a labels match matrix
    labels_eq = torch.eq(labels, labels.t()).float()

    # Positive distances (same labels)
    positive_distances = distance_matrix * labels_eq

    # Negative distances (different labels)
    negative_distances = distance_matrix * (1 - labels_eq)

    # Triplet loss calculation
    loss = torch.max(positive_distances - negative_distances + margin, torch.zeros_like(distance_matrix)).mean()
    return loss

