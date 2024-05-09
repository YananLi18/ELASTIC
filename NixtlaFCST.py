from torch.utils.data import DataLoader
import argparse
import time
from datetime import datetime
import sys
from util import EnsSDateset, metric, site_index
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, MSTL, SeasonalNaive, AutoTheta, TSB, ARCH, GARCH, ARIMA, ETS
from joblib import dump 
import os
# setting path
sys.path.append('../')
from utils import *
import warnings

# 忽略特定类型的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--site', type=str, default='AllSite', help='if valid the CloudOnlymodel in a specific site, pls replace "AllSite"!')
parser.add_argument('--model', type=str, default='ARIMA',
                    help='[ARIMA, Holt, Prophet]')
parser.add_argument('--data', type=str, default='All_full', help='data path')
parser.add_argument('--type', type=str, default='CPU', help='data type')
parser.add_argument('--freq', type=str, default='15min', help='frequency')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--label_len', type=int, default=0, help='informer’s start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
parser.add_argument('--chgpnt_scale', type=float, default=0.15, help='changepoint_prior_scale of Prophet')

parser.add_argument('--root_path', type=str, default='./data/', help='dragon:/data2/lyn/www23/data/')
parser.add_argument('--save', type=str, default='./garage/', help='save path')
parser.add_argument('--rate', type=float, default=2, help='network transmission rate[0.5, 2, 6.25]')
args = parser.parse_args()

type_parser = {
    'CPU': 'cpu_rate.csv',
    'UP': 'up_bw.csv',
    'DOWN': 'down_bw.csv'
}

data_path = './data/' + args.data + '/' + args.freq + '/'
testDataset = EnsSDateset(root_path='./data/', flag='test', size=[args.seq_length,args.pred_len, args.label_len], data=args.data,
                          data_type=args.type, scale=False, freq=args.freq)
test_dataloader = DataLoader(testDataset, batch_size=1)


models = [
    ARIMA(),
    AutoARIMA(), 
    AutoETS(), 
    AutoCES(), 
    MSTL(season_length=96),
    AutoTheta()
]
model_names = ['ARIMA','AutoARIMA','AutoETS','CES','MSTL','AutoTheta']

def trainmodels(test_dataloader, models, model_names):
    trues = []
    preds = {model_name: [] for model_name in model_names}
    mean_mae = {model_name: [] for model_name in model_names}
    mean_smape = {model_name: [] for model_name in model_names}
    mean_mse = {model_name: [] for model_name in model_names}
    mean_r2 = {model_name: [] for model_name in model_names}
    rate = args.rate
    val_time = []

    start = datetime.now()
    print("start train")
    for i, (x, y, _ , _) in enumerate(test_dataloader):
        size = x.element_size() * x.nelement()
        x = x.transpose(1, 3).squeeze().numpy()
        y = y.transpose(1, 3).squeeze().numpy()
        print("start one epoch")
        cnt = 0
        for row, true in zip(x, y):
            all_same = all(x == row[0] for x in row)
            all_zero = all(x == 0 for x in row)
            if (all_same == True) or (all_zero == True):
                continue
            dct = {'unique_id': 'CPU', 'ds': pd.date_range('2020-06-01', periods=len(row), freq=args.freq), 'y': row}
            df = pd.DataFrame(dct)
            fcst = StatsForecast(models=models, freq='15T', n_jobs=-1, fallback_model = SeasonalNaive(season_length=96))
            s1 = time.time()
            forecast = fcst.forecast(df=df,h=12)

            # time.sleep(size * 2 / 1024 / 1024 / rate)
            s2 = time.time()
            val_time.append(s2 - s1)
            trues.append(true)
            # true = np.array(true).reshape(-1)
            true = np.array(true)
            i_true = torch.from_numpy(true)
            for model_name in model_names:
                pred_model = forecast[model_name].values
                preds[model_name].append(pred_model)
                pred_model = np.array(pred_model)
                i_pred = torch.from_numpy(pred_model)
                i_mae, i_smape, i_mse, i_r2 = metric(i_pred, i_true)
                mean_mae[model_name].append(i_mae)
                mean_smape[model_name].append(i_smape)
                mean_mse[model_name].append(i_mse)
                mean_r2[model_name].append(i_r2)
        print("finished one epoch")
        
        if i % 10 == 0 and i != 0:
            print(f"epoch: {i} is processed with time {datetime.now()-start}:")
            for model_name in model_names:
                mmae = np.mean(mean_mae[model_name])
                msmape = np.mean(mean_smape[model_name])
                mmse = np.mean(mean_mse[model_name])
                mr2 = np.mean(mean_r2[model_name])
                print(f"Model: {model_name:10} | MAE: {mmae:7.4f} | SMAPE: {msmape:7.4f} | MSE: {mmse:7.4f} | R2: {mr2:7.4f}")

                mean_mae[model_name].clear()
                mean_smape[model_name].clear()
                mean_mse[model_name].clear()
                mean_r2[model_name].clear()


    print("Total Inference Time: {:.4f} secs".format(np.sum(val_time)))
    # trues = np.array(trues).reshape(-1)
    trues = np.array(trues)
    for model_name in model_names:
        preds[model_name]=np.array(preds[model_name])
    return  preds, trues

if args.site == 'AllSite':
    preds, trues = trainmodels(test_dataloader, models, model_names)

    save_path = args.save +'/NixtlaFCST/' + args.data + '/'+ args.type + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savez(save_path + 'preds.npz', **preds)
    np.save(save_path + 'trues.npy', trues)
    
else:
    vmindex_in_site = site_index(args.root_path, args.data, args.freq, args.site)
    save_path = args.save +'/NixtlaFCST/' + args.data + '/'+ args.type + '/'
    
    preds_loaded = np.load(save_path + 'preds.npz',allow_pickle=True)
    preds_loaded = dict(preds_loaded)

    R = preds_loaded[model_names[0]].shape[0]
    filtered_vmindex_in_site = [index for index in vmindex_in_site if index <= R]

    for model_name in model_names:
        preds_loaded[model_name]=preds_loaded[model_name][filtered_vmindex_in_site,:]
    preds = preds_loaded

    trues_loaded = np.load(save_path + 'trues.npy',allow_pickle=True)
    trues = trues_loaded[filtered_vmindex_in_site,:]

trues = torch.from_numpy(trues)
for model_name in model_names:
    #preds_model = np.array(preds[model_name]).reshape(-1)
    preds_model = preds[model_name]
    preds_model = torch.from_numpy(preds_model)
    amae = []
    amape = []
    armse = []
    ar2 = []
    print(f"Model: {model_name:<10}")
    for i in range(args.pred_len):
        mae, smape, mse, r2 = metric(preds_model[:,:i+1], trues[:,:i+1])
        print(f"horizon {i+1:4d}, MAE: {mae:<7.4f} | SMAPE: {smape:<7.4f} | MSE: {mse:<7.4f} | R2: {r2:<7.4f}")
        amae.append(mae)
        amape.append(smape)
        armse.append(mse)
        ar2.append(r2)
    print(f"On average over 12 horizons, MAE: {np.mean(amae):<7.4f} | SMAPE: {np.mean(amape):<7.4f} | MSE: {np.mean(armse):<7.4f} | R2: {np.mean(ar2):<7.4f}")  