import pandas as pd
from datetime import timedelta
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

parser = argparse.ArgumentParser(description='Padding')

parser.add_argument('--root_path', type=str, default='/data/zph/HGCN/data/', help='data path')
parser.add_argument('--region', type=str, default='All', help='which region (CC, EC, NC, NW, SC, SW)')
parser.add_argument('--gra', type=str, default='15min', help='Granularity of time series (5, 10, 15, 30, 1h)')
parser.add_argument("--period", type=int, default=96, help="the num of time unit to be a T")
parser.add_argument('--paint', action='store_true', default=False, help="paint the seasonal_decompose or not")
parser.add_argument("--output_dir", type=str, default="/data/zph/HGCN/data/", help="Output directory.")
args = parser.parse_args()


def find_continuous_zeros(data, min_length=pd.Timedelta(hours=4), max_length=pd.Timedelta(days=4)):
    zero_indices = data.index[data == 0]
    gaps = zero_indices.to_series().diff().ne(pd.Timedelta(minutes=15)).cumsum()
    continuous_zeros = zero_indices.to_series().groupby(gaps).agg(['first', 'last', 'count'])
    continuous_zeros['duration'] = continuous_zeros['last'] - continuous_zeros['first'] + pd.Timedelta(minutes=15)
    
    # 不过滤大于max_length的数据段
    return continuous_zeros[(continuous_zeros['duration'] >= min_length)]

def detect_stable_region(trend, window_size, threshold=0.0001):
    """
    Checks if there is a stable region in the trend data based on rate of change and a threshold.
    
    Args:
    - trend: The trend data.
    - window_size: The size of the sliding window for calculating average rate of change (Timedelta).
    - threshold: The threshold for the average rate of change.
    
    Returns:
    - has_stable: True if a stable region is detected, False otherwise.
    """
    if np.abs(trend.max()) > 100 * np.abs(trend.mean()):
        return True
    
    trend = trend.replace(0, np.nan).fillna(0)
    trend = trend.replace(0, np.nan).dropna()
    # Calculate the rate of change
    rate_of_change = np.abs(np.diff(trend) / trend[:-1])
    
    # Convert window size to integer number of data points
    window_size_points = int(window_size / pd.Timedelta(minutes=15))
   
    # Create a rolling window and calculate mean rate of change
    rolling_avg_rate = pd.Series(rate_of_change).rolling(window=window_size_points, min_periods=1).mean()
    # Check if any rolling average rate is below threshold
    if (rolling_avg_rate < threshold).any():
        return True
    else:
        return False

def padding_zero_data(pdfath, datatype):
    dataname = 'CPU'
    if datatype == 1:
        dataname = 'CPU'
    else :
        dataname = 'BW'
 
    df = pd.read_csv(pdfath)#合并变量形成具体路径，前者为路径名，后者为csv文件名
    df.fillna(0, inplace=True)
    df['date'] = pd.to_datetime(df['date'])#使用 pandas 库中的 pd.to_datetime 函数将数据帧 bw_raw 中名为 "date "的列转换为日期格式
    df.set_index('date', inplace=True)

    cnt = 0
    min_length=pd.Timedelta(hours=4)
    max_length=pd.Timedelta(days=4)

    for column in df.columns:
        # 跳过日期列
        if column == 'date':
            continue
        cnt += 1
        continuous_zeros_info = find_continuous_zeros(df[column])
        
        # 处理连续为零的数据段
        deleted = False
        result = seasonal_decompose(df[column], model='additive', period=args.period)
        trend = result.trend.fillna(0)
        seasonal = result.seasonal.fillna(0)
        resid = result.resid.fillna(0)
        for _, row in continuous_zeros_info.iterrows():
            start, end, count, duration = row
            # 删除持续时间超过max_length的时间序列
            if duration > max_length:
                df.drop(columns=[column], inplace=True)
                deleted = True
                break  # 跳出内层循环，处理下一个时间序列
            # 对于其他持续时间介于min_length和max_length之间的数据段，进行填充处理
            elif duration >= min_length:
                # 确定前后周期的时间点
                prev_cycle_idx = start - pd.Timedelta(days=1)
                next_cycle_idx = end + pd.Timedelta(days=1)

                # 往前找不缺失的周期
                while prev_cycle_idx in df[df[column] == 0].index and prev_cycle_idx > df.index[0]:
                    prev_cycle_idx -= pd.Timedelta(days=1)
                prev_cycle_idx -= pd.Timedelta(days=1)#在做残差均值时，需要保证一个完整周期的存在，所以保守起见再往前找一个周期

                # 往后找不缺失的周期
                while next_cycle_idx in df[df[column] == 0].index and next_cycle_idx < df.index[-1]:
                    next_cycle_idx += pd.Timedelta(days=1)
                next_cycle_idx += pd.Timedelta(days=1)#在做残差均值时，需要保证一个完整周期的存在，所以保守起见再往后找一个周期

                # 如果next_cycle_idx是结束索引且值仍为0，删除这个时间序列
                if (next_cycle_idx == df.index[-1] and df.loc[next_cycle_idx, column] == 0) \
                    or next_cycle_idx > df.index[-1] :
                    df.drop(columns=[column], inplace=True)
                    deleted = True
                    break  # 跳出循环
                # 如果prev_cycle_idx是起始索引且值仍为0，使用next_cycle_idx的值填充
                if (prev_cycle_idx == df.index[0] and df.loc[prev_cycle_idx, column] == 0) \
                    or prev_cycle_idx < df.index[0]:
                    prev_cycle_idx = next_cycle_idx# 让前一个周期等于后一个周期，然后继续执行填充

                #过滤噪声数据
                fistdate=pd.to_datetime("2020-06-01 12:00:00")
                isNoise=detect_stable_region(trend[fistdate:prev_cycle_idx], window_size=pd.Timedelta(days=4))

                if isNoise or np.abs(trend[fistdate:].max()) > 20 * np.abs(trend[fistdate:].mean()):
                    df.drop(columns=[column], inplace=True)
                    deleted = True
                    print(f"Noise:{column}")
                    break  # 跳出循环

                # 1-趋势分量的线性插值
                num_points = int((end - start).total_seconds() // 900 + 1)
                trend_fill = np.linspace(trend.loc[prev_cycle_idx], trend.loc[next_cycle_idx], num_points)

                # 2-残差分量的均值插值
                # 计算填充区间内的时间点数量
                num_points = int((end - start).total_seconds() // 900 + 1)
                # 获取填充区间内每个时间点的索引
                fill_indices = pd.date_range(start, periods=num_points, freq='15T')
                # 使用前后周期的残差值
                resid_fill = []
                for i, idx in enumerate(fill_indices):
                    # 计算对应的前一个和后一个周期的时间点
                    prev_period_idx = prev_cycle_idx + pd.Timedelta(minutes=15 * i)
                    next_period_idx = next_cycle_idx - pd.Timedelta(days=1) + pd.Timedelta(minutes=15 * i)
                    # 获取对应周期的残差值，如果不存在则使用0
                    prev_resid = resid[prev_period_idx] if prev_period_idx in resid.index else 0
                    next_resid = resid[next_period_idx] if next_period_idx in resid.index else 0
                    # 将前后周期的残差值平均化
                    avg_resid = (prev_resid + next_resid) / 2
                    resid_fill.append(avg_resid)
                resid_fill = np.array(resid_fill)

                # 3-季节性分量的复制
                seasonal_fill = seasonal.reindex(fill_indices, method='ffill').to_numpy()

                # 合成新的时间序列数据点
                filled_values = trend_fill + seasonal_fill + resid_fill

                # 填充缺失值
                df.loc[start:end, column] = filled_values

        if deleted == True or args.paint == False:#如果删除了就跳过绘图阶段
            continue

        
        # 对填充后的数据进行季节性分解
        filled_result = seasonal_decompose(df[column], model='additive', period=4*24)

        # 定义日期格式，仅显示日
        date_format = mdates.DateFormatter('%d')
        # 设置刻度位置为每天
        day_locator = mdates.DayLocator()

        # 绘制填充后的数据、趋势、季节性和残差
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        filled_result.observed.plot(ax=axes[0], title='Filled Original')
        filled_result.trend.plot(ax=axes[1], title='Filled Trend')
        filled_result.seasonal.plot(ax=axes[2], title='Filled Seasonality')
        filled_result.resid.plot(ax=axes[3], title='Filled Residuals')

        # 设置横坐标日期格式
        for ax in axes:
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_major_locator(day_locator)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # 保存图像
        output = args.root_path+args.region+'/'+args.gra+'/fill_and_filter'+ dataname +'/'
        if not os.path.exists(output):
            os.makedirs(output)
        plt.savefig(f'{output}prd{args.period}_{cnt}_paded.png')
        print(f"paint filled seasonal decompose of series :{cnt},{column}")

        # 关闭当前绘图窗口，避免重叠
        plt.close()

    return df

cpu_path = args.root_path+args.region+'/'+args.gra+'/'+'cpu_rate.csv'
cpu = padding_zero_data(cpu_path,1)
vmlist_c = cpu.columns.tolist()

bw_path = args.root_path+args.region+'/'+args.gra+'/'+'up_bw.csv'
bw = padding_zero_data(bw_path,2)
vmlist_u = bw.columns.tolist()

if vmlist_u != vmlist_c:#只保留两列数据都有的vm
    vmlist = [x for x in vmlist_u if x in vmlist_c]
    all_ins_u = bw[vmlist]
    all_ins_c = cpu[vmlist]
else:
    vmlist = vmlist_u

folder_path = args.output_dir + args.region + '_full' + '/' + args.gra + '/'
print("output path", folder_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
all_ins_u.to_csv(folder_path + 'up_bw.csv')
all_ins_c.to_csv(folder_path + 'cpu_rate.csv')
print(f"{folder_path}-----{len(vmlist)} VMs have complete data")
vm_d = pd.DataFrame(vmlist, columns=['vm'])
vm_d.to_csv(folder_path + 'vmlist.csv', index=False)