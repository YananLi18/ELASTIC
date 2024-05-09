import argparse
import time

import torch

from engine import *
import os
import shutil
import random
import pandas as pd
import numpy as np
from util import Dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='')
parser.add_argument('--model', type=str, default='gwnet', help='adj type (gwnet)')
parser.add_argument('--type', type=str, default='CPU', help='[CPU, UP]')
parser.add_argument('--freq', type=str, default='15min', help='frequency')
parser.add_argument('--data', type=str, default='CC', help='[site, NC, NW, ...]')
parser.add_argument('--site', type=str, default='nanjing-cmcc', help='')
parser.add_argument('--g_adjtype', type=str, default='all1', help='adj type of global adj [all1,dis,tem,rtt,identity]')
parser.add_argument('--adjtype', type=str, default='all', help='adj type of local adj [all, phy, log, identity]')
parser.add_argument('--addaptadj', action='store_false', default=True, help='whether add adaptive adj for local stage')
parser.add_argument('--g_addaptadj', action='store_false', default=True, help='whether add adaptive adj for whole stage')
parser.add_argument('--rate', type=float, default=6.25, help='network transmission rate[0.5, 2, 6.25]')
parser.add_argument('--contrastive', type=str, default='RNC', help='contrastive loss function[RNC,NT-Xent,NT-Logistic,Margin-Triplet]')


parser.add_argument('--dis_thre', type=float, default=4e5, help='adj type')
parser.add_argument('--rtt_thre', type=float, default=30, help='adj type')
parser.add_argument('--temp_thre', type=float, default=40, help='adj type')

parser.add_argument('--gat', action='store_true', default=False, help="whether replace gcn with gat")
parser.add_argument('--scaler', action='store_true', default=False, help='whether add scaler')
parser.add_argument('--sever', action='store_true', default=True, help='whether run in sever')
parser.add_argument('--glob_m', action='store_true', default=True, help='whether use global model')
parser.add_argument('--disagg_type',type=int, default=2, help='which kind of disagg on local model output [1,2]')
# parser.add_argument('--data', type=str, default='data/XiAn_City', help='data path')

parser.add_argument('--root_path', type=str, default='./data/')
parser.add_argument('--adjdata', type=str, default='data/XiAn_City/adj_mat.pkl', help='not use')

parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--label_len', type=int, default=0, help='transformer start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
# parser.add_argument('--num_nodes',type=int,default=792,help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=50, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--force', action='store_true', default=True, help="remove params dir")
parser.add_argument('--save', type=str, default='./garage/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
args = parser.parse_args()

seed = 3407 #42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def iteronmodel(g_dataloader, dataloader, idx, gmodel, device):#两个数据加载器g_dataloader和dataloader、一个索引张量idx、一个模型gmodel和一个设备device。
    lst = []
    for (x, y, _, _), (x1, y1, _, _) in zip(g_dataloader.get_iterator(), dataloader.get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)

        testx1 = torch.Tensor(x1).to(device, dtype=torch.float)
        testx1 = testx1.transpose(1, 3)
        testy1 = torch.Tensor(y1).to(device, dtype=torch.float)
        testy1 = testy1.transpose(1, 3)
        with torch.no_grad():
            if args.glob_m:#gwnet_lg 根据testx1来规范和格式化testx的形态。
                preds,_,_ = gmodel.model(testx, testx1, testy1[:, 0, :, :])  # output = [batch_size, 12, num_nodes, 1]
            else:
                preds = torch.Tensor(y).to(device)#预测结果已经在 `y` 中
            lst.append(torch.index_select(preds, 2, idx))#从每批预测中选择一个子集。这种选择可能会用于提取特定节点的预测结果。 按第三维输出行，输出idx索引的行
    return lst


def main():#局部完整阶段 融合全局时空模型和局部时空模型
    if not args.sever:
        args.root_path = './data/'
    device = torch.device('cuda:' + args.device)

    # Global mode init
    g_vmlist = pd.read_csv(args.root_path + 'site/' + args.freq + '/' + 'vmlist.csv')
    g_sensor_num = len(g_vmlist)
    g_adj_mx = util.global_adj_(args.root_path + 'site/' + args.freq + '/', args.type, adjtype=args.g_adjtype,
                                dis_threshold=args.dis_thre, rtt_threshold=args.rtt_thre,
                                temp_threshold=args.temp_thre, flag=False)
    g_dataloader = util.load_dataset(args.root_path, args.type, 'site', args.batch_size,
                                     args.seq_length, args.pred_len, args.scaler)
    g_supports = [torch.tensor(i).float().to(device) for i in g_adj_mx]
    g_scaler = g_dataloader['scaler'] if args.scaler else None
    # g_engine = trainer1(g_scaler, args.in_dim, args.seq_length, g_sensor_num, args.nhid, args.dropout,
    # args.learning_rate, args.weight_decay, device, g_supports, args.decay, args.gat, args.addaptadj)

    # Local model init
    vmlist = pd.read_csv(args.root_path + args.data + '/' + args.freq + '/' + 'vmlist.csv')
    ins = pd.read_csv(args.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores'])
    vmr = pd.merge(vmlist, ins, how='left', left_on='vm', right_on='uuid')
    idx = (vmr['ens_region_id'].values == args.site)
    sensor_num = sum(idx)
    idx_v = g_vmlist[g_vmlist['ens_region_id'] == args.site].index.to_list()[0]
    print(f"There are {sensor_num} VMs in {args.site} of {args.data}")
    adj_mx = util.local_adj_(args.root_path, args.data, args.freq, args.adjtype)
    vmindex_in_site = util.site_index(args.root_path, args.data, args.freq, args.site)
    vmindex_in_site = torch.tensor(vmindex_in_site)
    dataloader = util.load_dataset(args.root_path, args.type, args.data, args.batch_size,
                                   args.seq_length, args.pred_len, args.scaler, ratio_flag=False, site=args.site)
    scaler = dataloader['scaler'] if args.scaler else None
    supports = [torch.tensor(i[idx, :][:, idx]).float().to(device) for i in adj_mx]
    #supports = None
    g_engine = trainer_global(g_scaler, args.in_dim, args.seq_length, args.pred_len, g_sensor_num, args.nhid, args.dropout,
                              args.learning_rate, args.weight_decay, device, g_supports,supports, args.decay,
                              sensor_num, idx_v, args.gat, args.g_addaptadj, args.type, args.contrastive)
    if args.model == 'gwnet':
        engine = trainer0(scaler, args.in_dim, args.seq_length, args.pred_len, sensor_num, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay, args.data, args.site,
                          args.gat, args.addaptadj, args.type, args.disagg_type)
    elif args.model == 'LSTM':
        engine = trainer_lyn_lstm(scaler, args.in_dim, args.seq_length, args.pred_len, sensor_num, args.nhid, args.dropout,
                                  args.learning_rate, args.weight_decay, device, supports, args.decay, args.data,
                                  args.site, args.gat, args.addaptadj, args.type, args.disagg_type)
    elif args.model == 'OTSGGCN':
        engine = trainer_lyn_OTSGGCN(scaler, args.in_dim, args.seq_length,args.pred_len, sensor_num, args.nhid, args.dropout,
                                     args.learning_rate, args.weight_decay, device, supports, args.decay, args.data,
                                     args.site, args.gat, args.addaptadj, args.type, args.disagg_type)
    elif args.model == 'Gated_STGCN':
        engine = trainer_lyn_STGCN(scaler, args.in_dim, args.seq_length, sensor_num, args.nhid, args.dropout,
                                   args.learning_rate, args.weight_decay, device, supports, args.decay, args.data,
                                   args.site, args.gat, args.addaptadj, args.type, args.disagg_type)
    elif args.model == 'GRCN':
        engine = trainer_lyn_GRCN(scaler, args.in_dim, args.seq_length, sensor_num, args.nhid, args.dropout,
                                  args.learning_rate, args.weight_decay, device, supports, args.decay, args.data,
                                  args.site, args.gat, args.addaptadj, args.type, args.disagg_type)
    elif args.model == 'ASTGCN':
        engine = trainer_lyn_ASTGCN(scaler, args.in_dim, args.seq_length, sensor_num, args.nhid, args.dropout,
                                    args.learning_rate, args.weight_decay, device, supports, args.decay, args.data,
                                    args.site, args.gat, args.addaptadj, args.type, args.disagg_type)
    elif args.model == 'Autoformer':
        engine = trainer_lyn_Autoformer(scaler, args.in_dim, args.seq_length, args.pred_len, sensor_num, args.nhid, args.dropout,
                                    args.learning_rate, args.weight_decay, device, supports, args.decay, args.data,
                                    args.site, args.gat, args.addaptadj, args.type, args.disagg_type)
    elif args.model == 'TimesNet':
        engine = trainer_lyn_TimesNet(scaler, args.in_dim, args.seq_length, args.pred_len, sensor_num, args.nhid, args.dropout,
                                    args.learning_rate, args.weight_decay, device, supports, args.decay, args.data,
                                    args.site, args.gat, args.addaptadj, args.type, args.disagg_type)
    print(args)
    # print(engine.model.state_dict()['final_linear.weight'])
    # print(engine.model.state_dict()['W'].data)
    # check parameters file
    params_path = args.save + args.model + '/whole/' + args.data + '/' + args.site + '/' + args.type + '/' + args.freq \
                  + '/' + args.adjtype
    if os.path.exists(params_path) and not args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))
    #加载一个预先训练好的全局模型，并用它来转换数据,用转换后的数据创建新的数据加载器
    # Global model load pretrain best-performance model
    idx2 = torch.tensor(g_vmlist.index[g_vmlist['ens_region_id'] == args.site].tolist() * sensor_num).to(device)
    # g_engine.model.load_state_dict(torch.load("./model/" + "gwnet_site_CPU_0.22016.pth"))
    g_engine.model.load_state_dict(torch.load("./garage/gwnet/site2global/"
                                              + args.data + '/' + args.site + '/' + args.type + '/' + "best.pth"
                                              , map_location=device), strict=False) #加载预训练模型的状态字典
    g_engine.model.eval()
    #使用加载的模型来转换来自数据加载器g_dataloader的数据，并将结果连接到 NumPy 数组
    train_zs = torch.cat(iteronmodel(g_dataloader['train_loader'], dataloader['train_loader'],
                                     idx2, g_engine, device), dim=0).cpu().detach().numpy()
    val_zs = torch.cat(iteronmodel(g_dataloader['val_loader'], dataloader['val_loader'],
                                   idx2, g_engine, device), dim=0).cpu().detach().numpy()
    test_zs = torch.cat(iteronmodel(g_dataloader['test_loader'], dataloader['test_loader'],
                                    idx2, g_engine, device), dim=0).cpu().detach().numpy()
    #使用转换后的数据数组创建`assit_train_loader`、`assit_val_loader`和`assit_test_loader`。这些加载器用于模型另一部分的训练、验证和测试

    assit_train_loader = Dataloader(train_zs, np.zeros_like(train_zs), np.zeros(train_zs.shape[:2]), np.zeros(train_zs.shape[:2]), args.batch_size)
    assit_val_loader = Dataloader(val_zs, np.zeros_like(val_zs), np.zeros(val_zs.shape[:2]), np.zeros(val_zs.shape[:2]), args.batch_size)
    assit_test_loader = Dataloader(test_zs, np.zeros_like(test_zs), np.zeros(test_zs.shape[:2]), np.zeros(test_zs.shape[:2]), args.batch_size)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    transmit_bytes = 0.0
    rate = args.rate
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_r2 = []
        t1 = time.time()
        permutation = np.random.permutation(dataloader['train_loader'].size)
        xs, ys = dataloader['train_loader'].xs[permutation], dataloader['train_loader'].ys[permutation]
        xs1 = assit_train_loader.xs[permutation]
        dataloader['train_loader'].xs = xs
        dataloader['train_loader'].ys = ys
        assit_train_loader.xs = xs1
        # x in shape [batch_size, seq_len, num_nodes, 1]
        # y in shape [batch_size, pre_len, num_nodes, 1]
        for (x, y,  x_mark, y_mark), (x1, y1, _, _) in zip(dataloader['train_loader'].get_iterator(),
                                    assit_train_loader.get_iterator()):
            trainx = torch.Tensor(x).to(device, dtype=torch.float)
            trainy = torch.Tensor(y).to(device, dtype=torch.float)
            ass_ts = torch.Tensor(x1).to(device, dtype=torch.float)
            
            transmit_bytes += 3072 * 2
            time.sleep(3072 * 2 / 1024 / 1024 / rate)

            if args.model == 'Autoformer' or args.model == 'TimesNet':
                trainx_mark = torch.Tensor(x_mark).to(device, dtype=torch.float)#[64,12,5]
                trainy_mark = torch.Tensor(y_mark).to(device, dtype=torch.float)#[64,24,5]
                trainx=trainx[:, :, :, 0]
                trainy=trainy[:, :, :, 0]
                dec_inp = torch.zeros([ trainy.shape[0], args.pred_len, trainy.shape[-1]]).float().to(device)#
                dec_inp = torch.cat([ trainy[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
                metrics = engine.train(trainx, trainx_mark, trainy, dec_inp, trainy_mark, ass_ts)
            else:
                trainy = trainy.transpose(1, 3)
                trainx = trainx.transpose(1, 3)
                metrics = engine.train(trainx, trainy, ass_ts)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            train_r2.append(metrics[4])

        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []
        valid_r2 = []
        s1 = time.time()
        for (x, y, x_mark, y_mark), (x1, y1, _, _) in zip(dataloader['val_loader'].get_iterator(),
                                    assit_val_loader.get_iterator()):
            testx = torch.Tensor(x).to(device, dtype=torch.float)
            testy = torch.Tensor(y).to(device, dtype=torch.float)
            ass_ts = torch.Tensor(x1).to(device, dtype=torch.float)
            transmit_bytes += 3072 * 2
            time.sleep(3072 * 2 / 1024 / 1024 / rate)
            
            if args.model == 'Autoformer' or args.model == 'TimesNet':
                testx_mark = torch.Tensor(x_mark).to(device, dtype=torch.float)
                testy_mark = torch.Tensor(y_mark).to(device, dtype=torch.float)
                testx=testx[:, :, :, 0]
                testy=testy[:, :, :, 0]# __read_data__里多加了一维度,我们要去掉
                # decoder input
                dec_inp = torch.zeros([ testy.shape[0], args.pred_len, testy.shape[-1]]).float().to(device)
                dec_inp = torch.cat([ testy[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
                metrics = engine.eval(testx, testx_mark, testy, dec_inp, testy_mark, ass_ts)# batch_x, batch_x_mark, dec_inp, batch_y_mark
            else:
                testy = testy.transpose(1, 3)
                testx = testx.transpose(1, 3)
                metrics = engine.eval(testx, testy, ass_ts)
            
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
            valid_r2.append(metrics[4])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_r2 = np.mean(train_r2)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_r2 = np.mean(valid_r2)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Learning Rate: {}, Train MAE: {:.4f}, Train SMAPE: {:.4f}, Train MSE: {:.4f},Train R^2: {:.4f}\
                              Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid SMAPE: {:.4f}, Valid MSE: {:.4f}, Valid R^2: {:.4f},Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, engine.optimizer.param_groups[0]['lr'], mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_r2, mvalid_loss, mvalid_mae, mvalid_mape,
                         mvalid_rmse, mvalid_r2, (t2 - t1)), flush=True)
        
        torch.save(engine.model.state_dict(),
                   params_path + "/" + args.model + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Total Communication cost: {:.4f} MB".format(transmit_bytes / 1024 / 1024))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(params_path + "/" + args.model + "_epoch_" + str(bestid + 1) +
                                            "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    engine.model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # [batch_size, num_nodes, pred_len]
    for (x, y, x_mark, y_mark), (x1, y1, _, _) in zip(dataloader['test_loader'].get_iterator(),
                                assit_test_loader.get_iterator()):
        testx = torch.Tensor(x).to(device)
        ass_ts = torch.Tensor(x1).to(device, dtype=torch.float)

        if args.model == 'Autoformer' or args.model == 'TimesNet':
            testy = torch.Tensor(y).to(device)
            testx_mark = torch.Tensor(x_mark).to(device)
            testy_mark = torch.Tensor(y_mark).to(device)
            testx=testx[:, :, :, 0]
            testy=testy[:, :, :, 0]
        else :
            testx = testx.transpose(1, 3)

        with torch.no_grad():
            if args.model == 'Autoformer' or args.model == 'TimesNet':
                dec_inp = torch.zeros([ testy.shape[0], args.pred_len, testy.shape[-1]]).float().to(device)
                dec_inp = torch.cat([ testy[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
                preds = engine.model(testx, testx_mark, dec_inp, testy_mark, ass_ts)
                preds = preds.transpose(1, 3)   
            else :
                preds = engine.model(testx, ass_ts)
                preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    ar2 = []
    prediction = yhat
    for i in range(args.pred_len):
        # pred = prediction[:, :, :i+1]
        pred = scaler.inverse_transform(yhat[:, :, :i + 1]) if args.scaler else yhat[:, :, :i + 1]
        # prediction.append(pred)
        real = realy[:, :, :i + 1]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test SMAPE: {:.4f}, Test MSE: {:.4f}, Test R^2: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        ar2.append(metrics[3])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test SMAPE: {:.4f}, Test MSE: {:.4f}, Test R^2: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(ar2)))
    print(f"There are {sensor_num} VMs in {args.site} of {args.data}")
    # print(engine.model.state_dict()['final_linear.weight'])
    # print(engine.model.state_dict()['W'].data)
    torch.save(engine.model.state_dict(), params_path + "/" + args.model + "_exp" + str(args.expid) + "_best_" + str(
        round(his_loss[bestid], 2)) + ".pth")
    prediction_path = params_path + "/" + args.model + "_prediction_results"
    ground_truth = realy.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    # spatial_at = spatial_at.cpu().detach().numpy()
    # parameter_adj = parameter_adj.cpu().detach().numpy()
    np.savez_compressed(
        os.path.normpath(prediction_path),
        prediction=prediction,
        # spatial_at=spatial_at,
        # parameter_adj=parameter_adj,
        ground_truth=ground_truth
    )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
