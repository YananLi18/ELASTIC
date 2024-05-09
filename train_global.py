import argparse
import time
from engine import *
import os
import shutil
import random
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='')
parser.add_argument('--model', type=str, default='gwnet', help='adj type')
parser.add_argument('--type', type=str, default='CPU', help='[CPU, UP]')
parser.add_argument('--freq', type=str, default='15min', help='frequency')
parser.add_argument('--data', type=str, default='All_full', help='Directory where the data is located')
parser.add_argument('--site', type=str, default='nanjing-cmcc', help='')
parser.add_argument('--g_adjtype', type=str, default='all1', help='adj type of global adj [all1,dis,tem,rtt,identity]')
parser.add_argument('--adjtype', type=str, default='all', help='adj type of local adj [all, phy, log, identity]')
parser.add_argument('--addaptadj', action='store_false', default=True, help='whether add adaptive adj for global stage')
parser.add_argument('--contrastive', type=str, default='RNC', help='contrastive loss function[RNC,NT-Xent,NT-Logistic,Margin-Triplet]')

parser.add_argument('--rate', type=float, default=2, help='network transmission rate[0.5, 2, 6.25]')

parser.add_argument('--dis_thre', type=float, default=4e5, help='adj type')
parser.add_argument('--rtt_thre', type=float, default=30, help='adj type')
parser.add_argument('--temp_thre', type=float, default=40, help='adj type')

parser.add_argument('--gat', action='store_true', default=False, help="whether replace gcn with gat")
parser.add_argument('--scaler', action='store_true', default=False, help='whether add scaler')
parser.add_argument('--sever', action='store_true', default=True, help='whether run in sever')
# parser.add_argument('--data', type=str, default='data/XiAn_City', help='data path')

parser.add_argument('--root_path', type=str, default='./data/')
parser.add_argument('--adjdata', type=str, default='data/XiAn_City/adj_mat.pkl', help='adj data path')

parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
# parser.add_argument('--num_nodes',type=int,default=792,help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=50, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--force', action='store_true', default=True, help="remove params dir")
parser.add_argument('--save', type=str, default='./garage/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
args = parser.parse_args()

seed =3407#  42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():#The global phase trains the aggregation layer for each edge node.
    if not args.sever:
        args.root_path = './data/'
    device = torch.device('cuda:' + args.device)
    # Global model init
    g_vmlist = pd.read_csv(args.root_path + 'site/' + args.freq + '/' + 'vmlist.csv')
    g_sensor_num = len(g_vmlist)
    g_adj_mx = util.global_adj_(args.root_path + 'site/' + args.freq + '/', args.type, adjtype=args.g_adjtype,
                                dis_threshold=args.dis_thre, rtt_threshold=args.rtt_thre,
                                temp_threshold=args.temp_thre, flag=False)
    g_dataloader = util.load_dataset(args.root_path, args.type, 'site', args.batch_size,
                                     args.seq_length, args.pred_len, args.scaler)
    g_supports = [torch.tensor(i).float().to(device) for i in g_adj_mx]
    g_scaler = g_dataloader['scaler'] if args.scaler else None


    adj_mx = util.local_adj_(args.root_path, args.data, args.freq, args.adjtype)
    l_supports = [torch.tensor(i).float().to(device) for i in adj_mx]
    vmindex_in_site = util.site_index(args.root_path, args.data, args.freq, args.site)
    vmindex_in_site = torch.tensor(vmindex_in_site)
    l_supports = [mx[vmindex_in_site[:,None],vmindex_in_site] for mx in l_supports]

    # Local model init
    vmlist = pd.read_csv(args.root_path + args.data + '/' + args.freq + '/' + 'vmlist.csv')
    ins = pd.read_csv(args.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores'])
    vmr = pd.merge(vmlist, ins, how='left', left_on='vm', right_on='uuid')
    idx = (vmr['ens_region_id'].values == args.site)
    sensor_num = sum(idx)  
    idx_v = g_vmlist[g_vmlist['ens_region_id'] == args.site].index.to_list()[0]
    print(f"There are {sensor_num} VMs in {args.site} of {args.data}")
    l_dataloader = util.load_dataset(args.root_path, args.type, args.data, args.batch_size, args.seq_length,
                                     args.pred_len, args.scaler, ratio_flag=False, site=args.site)
    engine = trainer_global(g_scaler, args.in_dim, args.seq_length,args.pred_len, g_sensor_num, args.nhid, args.dropout,
                            args.learning_rate, args.weight_decay, device, g_supports,l_supports, args.decay,
                            sensor_num, idx_v, args.gat, args.addaptadj, args.type, args.contrastive)
    print(args)
    # check parameters file
    params_path = args.save + args.model + '/whole/' + args.data + '/' + args.site + '/' + args.type + '/' + args.freq
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
    transmit_bytes = 0.0
    rate = args.rate
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_r2 = []
        t1 = time.time()
       
        permutation = np.random.permutation(g_dataloader['train_loader'].size)
        xg, yg = g_dataloader['train_loader'].xs[permutation], g_dataloader['train_loader'].ys[permutation]
        g_dataloader['train_loader'].xs = xg
        g_dataloader['train_loader'].ys = yg
        
        xl, yl = l_dataloader['train_loader'].xs[permutation], l_dataloader['train_loader'].ys[permutation]
        l_dataloader['train_loader'].xs = xl
        l_dataloader['train_loader'].ys = yl

        # x in shape [batch_size, seq_len, num_nodes, 1]
        # y in shape [batch_size, pre_len, num_nodes, 1]
        for (x, y,_ ,_ ), (x1, y1,_ ,_ ) in zip(g_dataloader['train_loader'].get_iterator(),
                                    l_dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device, dtype=torch.float)
            trainx = trainx.transpose(1, 3)
            trainx1 = torch.Tensor(x1).to(device, dtype=torch.float)
            trainx1 = trainx1.transpose(1, 3)

            trainy = torch.Tensor(y).to(device, dtype=torch.float)
            trainy = trainy.transpose(1, 3)
            trainy1 = torch.Tensor(y1).to(device, dtype=torch.float)
            trainy1 = trainy1.transpose(1, 3)
            transmit_bytes += 3072 * 3
            time.sleep(3072 * 3 / 1024 / 1024 / rate)
            metrics = engine.train(trainx, trainx1, trainy[:, 0, :, :], trainy1[:, 0, :, :])
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

        for (x, y, _, _), (x1, y1, _, _) in zip(g_dataloader['val_loader'].get_iterator(),
                                    l_dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device, dtype=torch.float)
            testx = testx.transpose(1, 3)
            testx1 = torch.Tensor(x1).to(device, dtype=torch.float)
            testx1 = testx1.transpose(1, 3)

            testy = torch.Tensor(y).to(device, dtype=torch.float)
            testy = testy.transpose(1, 3)
            testy1 = torch.Tensor(y1).to(device, dtype=torch.float)
            testy1 = testy1.transpose(1, 3)
            metrics = engine.eval(testx, testx1, testy[:, 0, :, :], testy1[:, 0, :, :])
            transmit_bytes += 3072 * 2
            time.sleep(3072 * 2 / 1024 / 1024 / rate)
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
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_r2, mvalid_loss, mvalid_mae, mvalid_mape,
                         mvalid_rmse, mvalid_r2, (t2 - t1)), flush=True)
        
        torch.save(engine.model.state_dict(),
                   params_path + "/" + args.model + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Total Communication cost: {:.4f} MB".format(transmit_bytes / 1024 / 1024))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(
        params_path + "/" + args.model + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    engine.model.eval()

    outputs = []
    grounds = []
    #realy = torch.Tensor(g_dataloader['y_test']).to(device)
    #realy = realy.transpose(1, 3)[:, 0, :, :]  # [batch_size, num_nodes, pred_len]

    for (x, y, _, _), (x1, y1, _, _) in zip(g_dataloader['test_loader'].get_iterator(),
                                l_dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device, dtype=torch.float)
        testx = testx.transpose(1, 3)
        
        testy = torch.Tensor(y).to(device, dtype=torch.float)
        testy = testy.transpose(1, 3)
        
        testx1 = torch.Tensor(x1).to(device, dtype=torch.float)
        testx1 = testx1.transpose(1, 3)

        testy1 = torch.Tensor(y1).to(device, dtype=torch.float)
        testy1 = testy1.transpose(1, 3)

        with torch.no_grad():
            preds,_,output_re = engine.model(testx, testx1, testy1[:, 0, :, :])
            preds = preds.transpose(1, 3)
            output_re = output_re.transpose(1, 3).cpu().detach().to(device)
            testy[:, :, idx_v, :] = output_re[:, :, 0, :]
        grounds.append(testy.squeeze())
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(grounds, dim=0)
    
    # yhat = yhat[:realy.size(0), ...]
    
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    ar2 = []
    prediction = yhat
    for i in range(args.pred_len):
        # pred = prediction[:, :, :i+1]
        pred = g_scaler.inverse_transform(yhat[:, :, :i + 1]) if args.scaler else yhat[:, :, :i + 1]
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
    save_path = args.save + args.model + '/site2global/' + args.data + '/' + args.site + '/' + args.type + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(engine.model.state_dict(), save_path + "best.pth")
    prediction_path = save_path + "prediction_results"
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
