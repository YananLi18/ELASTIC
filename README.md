# ELASTIC

## Introduction

ELASTIC introduces a novel cloud-edge collaborative approach for edge workload forecasting, addressing the inefficiencies of traditional methods by integrating inter-site correlations. This project features a global aggregation layer for enhanced time efficiency and a local disaggregation layer to improve accuracy. Tested with datasets from Alibaba ENS, ELASTIC demonstrates superior performance, reducing both time consumption and communication costs.

The code is built on https://github.com/guokan987/HGCN.

The full datasets of Alibaba ENS is available on https://github.com/xumengwei/EdgeWorkloadsTraces


## Requirements

This project requires the following Python libraries:

- python==3.8.8
- numpy
- pandas==1.2.5
- pytorch==1.8.1
- scikit-learn==1.3.2
- tslearn==0.6.2
- sktime
- statsforecast==1.4.0


## Installation

1、To install these dependencies, run the following command:

```bash
conda create -n <yourenvname> python=3.8.8
source activate <yourenvname>
conda install pytorch==1.8.1 python==3.8.8  torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge  
conda install pandas==1.2.5
conda install -c conda-forge tslearn=0.6.2 python==3.8.8
conda install scikit-learn==1.3.2
conda install -c conda-forge sktime python==3.8.8
conda install -c conda-forge statsforecast     
```

If you encounter the error `AttributeError: module 'numpy' has no attribute 'long'` when running the program, it may be due to an incompatible version of the `numba` library. To resolve this issue, please ensure that you have the correct version of `numba` installed. The required version is `0.57.1`.
```bash
conda install numba==0.57.1 -c conda-forge
```

2、You can also install the required Python packages with the following command:

```bash
pip install -r requirements.txt
```


## To run


### 1-1. Global phase - Train aggregation layer for each edge node
```shell
# CPU
python train_global.py --device 0 --model gwnet --type CPU --data All_full --site nanjing-cmcc

# TP
python train_global.py --device 0 --model gwnet --type UP --data All_full --site nanjing-cmcc

```

### 1-2. Local complete phase - Fuse globalST model and localST model
```shell
# CPU
python train_whole.py --device 0 --model gwnet --type CPU --data All_full --site nanjing-cmcc

# TP
python train_whole.py --device 0 --model gwnet --type UP --data All_full --site nanjing-cmcc

```

### 2. Cloud-only baselines
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

# All available models : [gwnet, ASTGCN, GRCN, Gated_STGCN, H_GCN_wh, OGCRNN, OTSGGCN, LSTM, GRU, Informer, Autoformer, N_BEATS, TimesNet, DCRNN, NHITS, DeepAR]


```

## Citation
If you find this repo useful, please cite our paper.

```
@inproceedings{li2023elastic,
  title={ELASTIC: edge workload forecasting based on collaborative cloud-edge deep learning},
  author={Li, Yanan and Yuan, Haitao and Fu, Zhe and Ma, Xiao and Xu, Mengwei and Wang, Shangguang},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={3056--3066},
  year={2023}
}
```
