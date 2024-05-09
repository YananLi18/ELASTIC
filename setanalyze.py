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

parser = argparse.ArgumentParser(description='Analyze')

parser.add_argument('--root_path', type=str, default='/data2/penghongzhao/HGCN/data/', help='data path')
parser.add_argument('--region', type=str, default='All', help='which region (CC, EC, NC, NW, SC, SW)')
parser.add_argument('--gra', type=str, default='15min', help='Granularity of time series (5, 10, 15, 30, 1h)')
parser.add_argument("--period", type=int, default=96, help="the num of time unit to be a T")

parser.add_argument('--type', type=str, default='CPU', help='data type [CPU,BW]')
args = parser.parse_args()

type_parser = {
    'CPU': 'cpu_rate.csv',
    'BW': 'up_bw.csv',
}


df = pd.read_csv(args.root_path+args.region+'/'+args.gra+'/'+type_parser[args.type])#合并变量形成具体路径，前者为路径名，后者为csv文件名
df.fillna(0, inplace=True)
df['date'] = pd.to_datetime(df['date'])#使用 pandas 库中的 pd.to_datetime 函数将数据帧 bw_raw 中名为 "date "的列转换为日期格式
df.set_index('date', inplace=True)

cnt = 0
for column in df.columns:
    # 跳过日期列
    if column == 'date':
        continue
    cnt += 1

    result = seasonal_decompose(df[column], model='additive', period=4*24)
    # 定义日期格式，仅显示日
    date_format = mdates.DateFormatter('%d')
    # 设置刻度位置为每天
    day_locator = mdates.DayLocator()
    # 绘制原始数据、趋势、季节性和残差
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    result.observed.plot(ax=axes[0], title='Original')
    result.trend.plot(ax=axes[1], title='Trend')
    result.seasonal.plot(ax=axes[2], title='Seasonality')
    result.resid.plot(ax=axes[3], title='Residuals')

    # 设置横坐标日期格式
    for ax in axes:
        ax.xaxis.set_major_formatter(date_format)
        # 设置x轴每个刻度的间隔
        ax.xaxis.set_major_locator(day_locator)
        # 旋转 x 轴的刻度标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 可能需要根据您的实际情况调整
    # 保存图像
    print(f"paint seasonal decompose of series :{cnt},{column}")
    output = args.root_path+args.region+'/'+args.gra+'/seasonal_decompose_'+ args.type +'/'
    if not os.path.exists(output):
        os.makedirs(output)
    plt.savefig(f'{output}prd{args.period}_{cnt}_org.png')

    # 关闭当前绘图窗口，避免重叠
    plt.close()