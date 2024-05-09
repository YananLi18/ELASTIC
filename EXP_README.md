
The code is built on https://github.com/guokan987/HGCN.

The full datasets of Alibaba ENS is available on https://github.com/xumengwei/EdgeWorkloadsTraces


# 1. 仅局部时空预测模型预测

```shell
# 
# CPU
python train_local.py --device 0 --model OTSGGCN --type CPU --data CC --site wuhan-telecom

# TP
python train_local.py --device 0 --model OTSGGCN --type UP --data CC --site wuhan-telecom

```

# 2-1. 全局阶段 训练每个边缘节点的 聚合层
```shell
# CPU
python train_global.py --device 0 --model gwnet --type CPU --data All_full --site nanjing-cmcc

# TP
python train_global.py --device 0 --model gwnet --type UP --data All_full --site nanjing-cmcc

```

# 2-2. 局部完整阶段 融合全局时空模型和局部时空模型
```shell
# CPU
python train_whole.py --device 0 --model gwnet --type CPU --data All_full --site nanjing-cmcc

# TP
python train_whole.py --device 0 --model gwnet --type UP --data All_full --site nanjing-cmcc

```

# 3. Cloud-only baselines
```shell
# ARIMA MSTL
python NixtlaFCST.py --type CPU --data All_full --site AllSite
python NixtlaFCST.py --type CPU --data All_full --site nanjing-cmcc

# LSTM
python train.py --device 0 --model LSTM --type CPU --data All_full  --site nanjing-cmcc

# gwnet
python train.py --device 0 --model gwnet --type CPU --data All_full  --site nanjing-cmcc

# N_BEATS
python train.py --device 0 --model N_BEATS --type CPU --data All_full  --site nanjing-cmcc

# TimesNet
python train.py --device 0 --model TimesNet --type CPU --data All_full  --site nanjing-cmcc

# [gwnet, ASTGCN, GRCN, Gated_STGCN, H_GCN_wh, OGCRNN, OTSGGCN, LSTM, GRU, Informer, Autoformer, N_BEATS, TimesNet, DCRNN, NHITS, DeepAR]


```

