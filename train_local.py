import argparse
import time
from engine import *
import os
import shutil
import random
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='5', help='')
parser.add_argument('--model', type=str, default='gwnet', help='adj type')
parser.add_argument('--type', type=str, default='UP', help='[CPU, UP]')
parser.add_argument('--freq', type=str, default='15min', help='frequency')
parser.add_argument('--data', type=str, default='NC', help='[site, NC, NW, ...]')
parser.add_argument('--site', type=str, default='beijing-telecom', help='')
parser.add_argument('--adjtype', type=str, default='all', help='adj type')
# parser.add_argument('--dis_thre', type=float, default=4e5, help='adj type')
# parser.add_argument('--rtt_thre', type=float, default=30, help='adj type')
# parser.add_argument('--temp_thre', type=float, default=40, help='adj type')
parser.add_argument('--gat', action='store_true', default=False, help="whether replace gcn with gat")
parser.add_argument('--addaptadj', action='store_true', default=True, help='whether add adaptive adj')
parser.add_argument('--scaler', action='store_true', default=False, help='whether add scaler')
parser.add_argument('--sever', action='store_true', default=True, help='whether run in sever')
# parser.add_argument('--data', type=str, default='data/XiAn_City', help='data path')

parser.add_argument('--root_path', type=str, default='./data/', help='dragon:/data2/lyn/www23/data/')
parser.add_argument('--adjdata', type=str, default='data/XiAn_City/adj_mat.pkl', help='adj data path')

parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--label_len', type=int, default=0, help='informer’s start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
# parser.add_argument('--num_nodes',type=int,default=792,help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.008, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=50, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--force', action='store_true', default=True, help="remove params dir")
parser.add_argument('--save', type=str, default='./garage/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
args = parser.parse_args()

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():#局部时空模型
    if not args.sever:
        args.root_path = './data/'
    device = torch.device('cuda:' + args.device)
    vmlist = pd.read_csv(args.root_path + args.data + '/' + args.freq + '/' + 'vmlist.csv')
    ins = pd.read_csv(args.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores'])
    vmr = pd.merge(vmlist, ins, how='left', left_on='vm', right_on='uuid')
    idx = (vmr['ens_region_id'].values == args.site)#在同一个边缘站点site管理下的本地VM的索引表

    args.num_nodes = sum(idx)
    adj_mx = util.local_adj_(args.root_path, args.data, args.freq, args.adjtype)
    dataloader = util.load_dataset(args.root_path, args.type, args.data, args.batch_size,
                                   args.seq_length, args.pred_len, args.scaler, ratio_flag=False, site=args.site,
                                   lable_len=args.label_len,model=args.model) # Informer
    scaler = dataloader['scaler'] if args.scaler else None
    supports = [torch.tensor(i[idx, :][:, idx]).float().to(device) for i in adj_mx]

    print(args)
    # seq_length whether change to pred_len
    if args.model == 'gwnet':
        engine = trainer1(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay, args.gat, args.addaptadj,
                          args.type, args.data)
    elif args.model == 'ASTGCN_Recent':
        engine = trainer2(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'GRCN':
        engine = trainer3(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'Gated_STGCN':
        engine = trainer4(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'H_GCN_wh':
        engine = trainer5(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'OGCRNN':
        engine = trainer8(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'OTSGGCN':
        engine = trainer9(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'LSTM':
        engine = trainer10(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'GRU':
        engine = trainer11(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'Informer':
        engine = trainer12(args.in_dim, args.seq_length, args.pred_len, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay)
    # check parameters file
    params_path = args.save + args.model + '/' + args.data + '/' + args.type + '/' + args.freq
    if os.path.exists(params_path) and not args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_r2 = []
        t1 = time.time()
        # dataloader['train_loader'].shuffle()
        # x in shape [batch_size, seq_len, num_nodes, 1]
        # y in shape [batch_size, pre_len, num_nodes, 1]
        for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device, dtype=torch.float)
            trainy = torch.Tensor(y).to(device, dtype=torch.float)
            if args.model == 'Informer':
                trainx_mark = torch.Tensor(x_mark).to(device, dtype=torch.float)#[64,12,5]
                trainy_mark = torch.Tensor(y_mark).to(device, dtype=torch.float)#[64,24,5]
                trainx=trainx[:, :, :, 0]# [64,12,66,1]
                trainy=trainy[:, :, :, 0]# [64,24,66]__read_data__里多加了一维度,我们要去掉
                # decoder input 【pred_len+labe_len是informer的y的必需品】
                dec_inp = torch.zeros([ trainy.shape[0], args.pred_len, trainy.shape[-1]]).float().to(device)#[64,12,66]
                dec_inp = torch.cat([ trainy[:,:args.label_len,:], dec_inp], dim=1).float().to(device)#[64,0:12+12,66] 
                #[64,12,66,1],[64,12,5],[64,12,66],[64,24,66],[64,12,5]
                metrics = engine.train(trainx, trainx_mark, trainy, dec_inp, trainy_mark)# batch_x, batch_x_mark, dec_inp, batch_y_mark
            else :
                trainx = trainx.transpose(1, 3)
                trainy = trainy.transpose(1, 3)
                metrics = engine.train(trainx, trainy[:, 0, :, :])
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
        for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device, dtype=torch.float)
            testy = torch.Tensor(y).to(device, dtype=torch.float)
            
            if args.model == 'Informer':
                testx_mark = torch.Tensor(x_mark).to(device, dtype=torch.float)
                testy_mark = torch.Tensor(y_mark).to(device, dtype=torch.float)
                testx=testx[:, :, :, 0]
                testy=testy[:, :, :, 0]# __read_data__里多加了一维度,我们要去掉
                # decoder input
                dec_inp = torch.zeros([ testy.shape[0], args.pred_len, testy.shape[-1]]).float().to(device)
                dec_inp = torch.cat([ testy[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
                metrics = engine.eval(testx, testx_mark, testy, dec_inp, testy_mark)# batch_x, batch_x_mark, dec_inp, batch_y_mark
            else :
                testx = testx.transpose(1, 3)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:, 0, :, :])
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

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train SMAPE: {:.4f}, Train MSE: {:.4f}, Train R^2: {:.4f}\
                              Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid SMAPE: {:.4f}, Valid MSE: {:.4f}, Valid R^2: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_r2, 
                            mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_r2, (t2 - t1)), flush=True)
        torch.save(engine.model.state_dict(),
                   params_path + "/" + args.model + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(
        params_path + "/" + args.model + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    engine.model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    if args.model == 'Informer':
        realy = realy[:, :, :, 0]
    else :
        realy = realy.transpose(1, 3)[:, 0, :, :]  # [batch_size, num_nodes, pred_len]

    for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        if args.model == 'Informer':
            testy = torch.Tensor(y).to(device)
            testx_mark = torch.Tensor(x_mark).to(device)
            testy_mark = torch.Tensor(y_mark).to(device)
            testx=testx[:, :, :, 0]
            testy=testy[:, :, :, 0]
        else:
            testx = testx.transpose(1, 3)
        with torch.no_grad():
            if args.model == 'Informer':
                dec_inp = torch.zeros([ testy.shape[0], args.pred_len, testy.shape[-1]]).float().to(device)
                dec_inp = torch.cat([ testy[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
                preds = engine.model(testx, testx_mark, dec_inp, testy_mark)
            else:
                preds = engine.model(testx)
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
        if args.model == 'Informer' :
            pred = scaler.inverse_transform(yhat[:, :i + 1, :]) if args.scaler else yhat[:, :i + 1, :]
            # prediction.append(pred)
            real = realy[:, :i + 1, :]
        else:
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
