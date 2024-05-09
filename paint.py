
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
fig, ax = plt.subplots(figsize=(6.4, 4.8))

y = [23.45,25.52,28.85,23.02,22.64,10.63,11.31,17.16,13.98,9.47,12.19,32.71,
                14.64,19.30,13.68,11.19,20.85,11.68,7.68 ]
x = [5.0543, 4.4701, 7.5717, 8.0585, 8.0144, 9.8788, 23.8673,
                    7.9768, 7.8901, 14.1609, 26.7783, 84.4119, 24.4878,
                    20.1580, 12.4587, 13.6630, 14.5502, 15.4818, 4.4281]
n = ["ARIMA", "MSTL", "DeepAR", "LSTM", "GRU", "N-BEATS", "N-HiTS",
         "Informer", "Autoformer", "TimesNet", "GWNET", "DCRNN", "ASTGCN",
         "GCRN", "STGCN", "HGCN", "OGCRNN", "OTSGGCN", "ELASTIC"]
#x = [1.41, 1.43, 2.65, 2.75, 4.90, 4.50, 3.99]
#y = [27.67, 19.27, 12.07, 11.70, 5.47, 8.08, 9.84]

# n = ['Holt', 'ARIMA', 'LSTM', 'GRU', 'GWNET', 'HGCN', 'STGCN']

# 节点分组
ell1_nodes = ["ARIMA", "MSTL", "DeepAR"]
ell2_nodes = ["LSTM", "GRU", "N-BEATS", "N-HiTS"]
ell4_nodes = ["Informer", "Autoformer", "TimesNet"]
ell3_nodes = ["GWNET","DCRNN", "ASTGCN", "GCRN", "STGCN", "HGCN", "OGCRNN", "OTSGGCN"]

txt_fs=11
marker_sz=115
for i in range(0,3):
    if i== 0:
        ax.annotate(n[i], (x[i]-1.3, y[i]-1.3), fontsize=txt_fs)
    elif i == 1:
        ax.annotate(n[i], (x[i]-1, y[i]-1.3), fontsize=txt_fs)
    else :
        ax.annotate(n[i], (x[i]-1.3, y[i]-1.3), fontsize=txt_fs)
for i in range(3,7):
    if i == 3:
        ax.annotate(n[i], (x[i]+0.2, y[i]+0.5), fontsize=txt_fs)
    if i == 4:
        ax.annotate(n[i], (x[i]+0.3, y[i]-1.0), fontsize=txt_fs)
    if i == 5:
        ax.annotate(n[i], (x[i]-1.5, y[i]-1.5), fontsize=txt_fs)
    if i == 6:
        ax.annotate(n[i], (x[i]-3, y[i]-1.3), fontsize=txt_fs)
for i in range(7,10):
    ax.annotate(n[i], (x[i]-2, y[i]-1.3), fontsize=txt_fs)
for i in range(10,len(x)-1):
    if i == 10:
        ax.annotate(n[i], (x[i]-1, y[i]+0.8), fontsize=txt_fs)
    elif i == 15:
        ax.annotate(n[i], (x[i]-1.5, y[i]+0.5), fontsize=txt_fs)
    else:
        ax.annotate(n[i], (x[i]+0.2, y[i]+0.15), fontsize=txt_fs)


ax.text(5, 9, 'Goal', style='italic', c='w',
        fontsize=15,  horizontalalignment='center',verticalalignment='center')
ax.fill_between(np.arange(3, 7, 0.01), 4, 10, color='r', alpha=0.5)

# ell1 = Ellipse(xy=(5,32), width=8, height=18, angle=0,
#                label='Classical Methods', zorder=-1)
# ax.add_artist(ell1)
# ell1.set_clip_box(ax.bbox)
# ell1.set_facecolor('#8E9BC3')

# ell2 = Ellipse(xy=(15,15), width=15, height=22, angle=45,
# label='DNN-based Methods', zorder=-1)
# ax.add_artist(ell2)
# ell2.set_clip_box(ax.bbox)
# ell2.set_facecolor('#A2D3DD')

# ell3 = Ellipse(xy=(18.5, 15), width=12, height=20, angle=68,
# label='Spatial-temporal Methods', zorder=0)
# ax.add_artist(ell3)
# ell3.set_clip_box(ax.bbox)
# ell3.set_facecolor('#F9F2B0')

# ell4 = Ellipse(xy=(10.5, 13.5), width=3.8, height=12.5, angle=45,
# label='Transformer Methods', zorder=-1)
# ax.add_artist(ell4)
# ell4.set_clip_box(ax.bbox)
# ell4.set_facecolor('#F5D0A9')

#ax.scatter(x, y, 25, c='k')
sc=plt.scatter([x[i] for i in range(len(x)) if n[i] in ell1_nodes], 
            [y[i] for i in range(len(y)) if n[i] in ell1_nodes],
            marker='o', label='ell1', s=marker_sz)
colors = sc.get_facecolor()
colors_hex = [mcolors.rgb2hex(color) for color in colors]
print(colors_hex[0])
sc=plt.scatter([x[i] for i in range(len(x)) if n[i] in ell2_nodes], 
            [y[i] for i in range(len(y)) if n[i] in ell2_nodes],
            marker='^', label='ell2',s=marker_sz)
colors = sc.get_facecolor()
colors_hex = [mcolors.rgb2hex(color) for color in colors]
print(colors_hex[0])
sc=plt.scatter([x[i] for i in range(len(x)) if n[i] in ell4_nodes], 
            [y[i] for i in range(len(y)) if n[i] in ell4_nodes],
            marker='s', label='ell4',s=marker_sz)
colors = sc.get_facecolor()
colors_hex = [mcolors.rgb2hex(color) for color in colors]
print(colors_hex[0])
sc=plt.scatter([x[i] for i in range(len(x)) if n[i] in ell3_nodes], 
            [y[i] for i in range(len(y)) if n[i] in ell3_nodes],
            marker='P', label='ell3',s=marker_sz)
colors = sc.get_facecolor()
colors_hex = [mcolors.rgb2hex(color) for color in colors]
print(colors_hex[0])

ax.scatter(4.4281, 7.68, marker_sz, c='#FFF38C', marker='*',)

ax.annotate('ELASTIC', (3.5, 6.3), fontsize=txt_fs)
ax.set_xlim(3,30)
ax.set_ylim(5,30)
ax.set_xlabel('Time Consumption (secs)', fontsize=15)
ax.set_ylabel('Mean Absolute Error', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid(alpha=0.5)
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label='Classical Methods'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e', markersize=8, label='DNN-based Methods'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c', markersize=8, label='Transformer Methods'),
    plt.Line2D([0], [0], marker='P', color='w', markerfacecolor='#d62728', markersize=8, label='Spatial-temporal Methods')
]
ax.legend(handles=legend_elements, fontsize=13)

fig.tight_layout()

fig.savefig('intro_contrast.pdf')