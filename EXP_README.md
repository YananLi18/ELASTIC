
The code is built on https://github.com/guokan987/HGCN.

The full datasets of Alibaba ENS is available on https://github.com/xumengwei/EdgeWorkloadsTraces

# 以vm为粒度，数据集是局部的
# 1. 仅局部时空预测模型预测


```shell
# 
# CPU
python train_local.py --device 0 --model OTSGGCN --type CPU --data CC --site wuhan-telecom

# BW
python train_local.py --device 0 --model OTSGGCN --type UP --data CC --site wuhan-telecom

```

# 2+3 是以vm为粒度，一个站点（局部）的聚合解聚合训练
# 2. 全局阶段 训练每个边缘节点的 聚合层
```shell
# CPU
python train_global.py --device 0 --model gwnet --type CPU --data All_full --site nanjing-cmcc

# BW
python train_global.py --device 0 --model gwnet --type UP --data All_full --site nanjing-cmcc

```

# 3. 局部完整阶段 融合全局时空模型和局部时空模型
```shell
# CPU
python train_whole.py --device 0 --model gwnet --type CPU --data All_full --site nanjing-cmcc

# BW
python train_whole.py --device 0 --model gwnet --type UP --data All_full --site nanjing-cmcc

```
# 是以vm为粒度，所有站点（整体）训练，
# --data site是表示以site为粒度进行训练
# --site xxx-xxx 是表示只在某个站点下验证整体的模型
# 4. Cloud-only baselines
```shell
# ARIMA
python baseline_darts.py --model ARIMA --type CPU --data All_full

# LSTM
python train.py --device 0 --model LSTM --type CPU --data All_full  --site nanjing-cmcc

# HGCN
python train.py --device 0 --model gwnet --type CPU --data All_full  --site nanjing-cmcc

# ASTGCN
python train.py --device 0 --model ASTGCN_Recent --type CPU --data All_full  --site nanjing-cmcc

# 替换model GRCN, Gated_STGCN, H_GCN_wh, OGCRNN, OTSGGCN

# DCRNN 
python dcrnn_train_pytorch.py --type CPU --data CC2 

```

