import argparse
import time
from engine import *
import os
import shutil
import random
import pandas as pd
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='6', help='')
parser.add_argument('--model', type=str, default='gwnet', help='adj type')
parser.add_argument('--type', type=str, default='UP', help='[CPU, UP]')
parser.add_argument('--freq', type=str, default='15min', help='frequency')
parser.add_argument('--data', type=str, default='All_full', help='Directory where the data is located')
parser.add_argument('--site', type=str, default='AllSite', help='if valid the CloudOnlymodel in a specific site, pls replace "AllSite"!')
parser.add_argument('--adjtype', type=str, default='identity', help='adj type')
parser.add_argument('--rate', type=float, default=2, help='network transmission rate[0.5, 2, 6.25]')

parser.add_argument('--dis_thre', type=float, default=4e5, help='adj type')
parser.add_argument('--rtt_thre', type=float, default=30, help='adj type')
parser.add_argument('--temp_thre', type=float, default=40, help='adj type')

parser.add_argument('--gat', action='store_true', default=False, help="whether replace gcn with gat")
parser.add_argument('--addaptadj', action='store_true', default=False, help='whether add adaptive adj')
parser.add_argument('--scaler', action='store_true', default=False, help='whether add scaler')
parser.add_argument('--sever', action='store_true', default=True, help='whether run in sever')


parser.add_argument('--root_path', type=str, default='./data/', help='dragon:/data2/lyn/www23/data/')
parser.add_argument('--adjdata', type=str, default='data/XiAn_City/adj_mat.pkl', help='adj data path')

parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--label_len', type=int, default=0, help='transformer start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
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

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    if not args.sever:
        args.root_path = './data/'
    
    device = torch.device('cuda:' + args.device)
    root_path = args.root_path + args.data + '/' + args.freq + '/'          
    vmlist = pd.read_csv(root_path + 'vmlist.csv')
    sensor_num = len(vmlist)
    args.num_nodes = sensor_num
    
    if args.site != 'AllSite':
        vmindex_in_site = util.site_index(args.root_path, args.data, args.freq, args.site)
        

    if args.data == 'site':
        adj_mx = util.global_adj_(root_path, args.type, args.adjtype, dis_threshold=args.dis_thre,
                                  rtt_threshold=args.rtt_thre, temp_threshold=args.temp_thre, flag=False)
    else:
        adj_mx = util.local_adj_(args.root_path, args.data, args.freq, args.adjtype)
    dataloader = util.load_dataset(args.root_path, args.type, args.data, args.batch_size,
                                   args.seq_length, args.pred_len, args.scaler,
                                   lable_len=args.label_len,model=args.model)
    scaler = dataloader['scaler'] if args.scaler else None
    supports = [torch.tensor(i).float().to(device) for i in adj_mx]
    # supports = None
    print(args)
    # seq_length whether change to pred_len
    if args.model == 'gwnet':
        engine = trainer1(scaler, args.in_dim, args.seq_length,args.pred_len, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay, args.gat, args.addaptadj,
                         args.type, args.data)
    elif args.model == 'ASTGCN':
        engine = trainer2(args.in_dim, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'GRCN':
        engine = trainer3(args.in_dim, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'Gated_STGCN':
        engine = trainer4(args.in_dim, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'H_GCN_wh':
        engine = trainer5(args.in_dim, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'OGCRNN':
        engine = trainer8(args.in_dim, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'OTSGGCN':
        engine = trainer9(args.in_dim, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'LSTM':
        engine = trainer10(args.in_dim, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'GRU':
        engine = trainer11(args.in_dim, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'Informer':
        engine = trainer12(args.num_nodes, args.seq_length, args.pred_len, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'Autoformer':
        engine = trainer13(args.num_nodes, args.seq_length, args.pred_len, args.label_len, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'N_BEATS':
        engine = trainer14(dataloader['train_loader'].len(), args.seq_length, args.pred_len, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'TimesNet':
        engine = trainer15(args.num_nodes, args.seq_length, args.pred_len, args.label_len, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'DCRNN':
        engine = trainer16(args.in_dim, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'NHITS':
        engine = trainer17(args.seq_length, args.pred_len, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay)
    elif args.model == 'DeepAR':
        engine = trainer18(args.batch_size, args.seq_length, args.pred_len, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay)
    # check parameters file
    params_path = args.save + args.model + '/' + args.data + '/' + args.type + '/' + args.freq + '/' +"Device"+args.device 
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
    for i in range(1, args.epochs+1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_r2 = []
        t1 = time.time()
       
        # x in shape [batch_size, seq_len, num_nodes, 1]
        # y in shape [batch_size, pre_len, num_nodes, 1]
        for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device, dtype=torch.float)
            trainy = torch.Tensor(y).to(device, dtype=torch.float)
            transmit_bytes += 3072 * 2 * args.num_nodes
            time.sleep(3072 * 2 * 48 / 1024 / 1024 / rate)
            if args.model == 'Informer' or args.model == 'Autoformer' or args.model == 'TimesNet':
                trainx_mark = torch.Tensor(x_mark).to(device, dtype=torch.float)
                trainy_mark = torch.Tensor(y_mark).to(device, dtype=torch.float)
                trainx=trainx[:, :, :, 0]
                trainy=trainy[:, :, :, 0]
                dec_inp = torch.zeros([ trainy.shape[0], args.pred_len, trainy.shape[-1]]).float().to(device)#
                dec_inp = torch.cat([ trainy[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
                metrics = engine.train(trainx, trainx_mark, trainy, dec_inp, trainy_mark)
            elif args.model == 'N_BEATS':
                trainx = trainx.transpose(1, 3)
                trainy = trainy.transpose(1, 3)
                metrics = engine.train(trainx, trainy, iter)
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
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []
        valid_r2 = []
        s1 = time.time()
        for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device, dtype=torch.float)
            testy = torch.Tensor(y).to(device, dtype=torch.float)
            transmit_bytes += 3072 * 2 * args.num_nodes
            time.sleep(3072 * 2 * 48 / 1024 / 1024 / rate)

            if args.model == 'Informer' or args.model == 'Autoformer' or args.model == 'TimesNet': 
                testx_mark = torch.Tensor(x_mark).to(device, dtype=torch.float)
                testy_mark = torch.Tensor(y_mark).to(device, dtype=torch.float)
                testx=testx[:, :, :, 0]
                testy=testy[:, :, :, 0]
                # decoder input
                dec_inp = torch.zeros([ testy.shape[0], args.pred_len, testy.shape[-1]]).float().to(device)
                dec_inp = torch.cat([ testy[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
                metrics = engine.eval(testx, testx_mark, testy, dec_inp, testy_mark)
            elif args.model == 'N_BEATS':
                testx = testx.transpose(1, 3)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy)
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
        print(log.format(i, (s2-s1)))
        val_time.append(s2-s1)                             
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
        
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Learning Rate: {:.4f}, Train MAE: {:.4f}, Train SMAPE: {:.4f}, Train MSE: {:.4f}, Train R^2: {:.4f} \
                              Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid SMAPE: {:.4f}, Valid MSE: {:.4f}, Valid R^2: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, engine.optimizer.param_groups[0]['lr'], mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_r2, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_r2, (t2 - t1)),flush=True)
        
        torch.save(engine.model.state_dict(), params_path+"/"+args.model+"_epoch_"+str(i)+"_"+str(round(mvalid_loss, 2)) + ".pth")
        his_loss.append(mvalid_loss)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Total Communication cost: {:.4f} MB".format(transmit_bytes / 1024 / 1024))

    #testing                                                                   
    bestid = np.argmin(his_loss)                                                
    engine.model.load_state_dict(torch.load(params_path+"/"+args.model+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
    engine.model.eval()                                                         
    torch.save(engine.model.state_dict(),
               "./"+"model/"+args.model+"_"+args.data+"_"+args.type+"_"+str(round(his_loss[bestid], 5))+".pth")

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :] # [batch_size, num_nodes, pred_len]

    for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        if args.model == 'Informer' or args.model == 'Autoformer' or args.model == 'TimesNet':
            testy = torch.Tensor(y).to(device)
            testx_mark = torch.Tensor(x_mark).to(device)
            testy_mark = torch.Tensor(y_mark).to(device)
            testx=testx[:, :, :, 0]
            testy=testy[:, :, :, 0]
        elif args.model == 'N_BEATS':
            testx = testx.transpose(1, 3)# [batch_size, 1, feature, pred_len]
        else:
            testx = testx.transpose(1, 3)
        with torch.no_grad():                                                   
            if args.model == 'Informer' or args.model == 'Autoformer' or args.model == 'TimesNet':
                dec_inp = torch.zeros([ testy.shape[0], args.pred_len, testy.shape[-1]]).float().to(device)
                dec_inp = torch.cat([ testy[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
                preds = engine.model(testx, testx_mark, dec_inp, testy_mark)
                preds = preds.transpose(1, 2)
            elif args.model == 'N_BEATS' or args.model == 'NHITS' or args.model == 'DeepAR':
                preds = engine.model(testx) # [batch_size, 1, feature, pred_len]
            elif args.model == 'DCRNN':
                preds = engine.model(testx[:, 0, :, :],testy.transpose(1, 3)[:, 0, :, :])     #[b,num_nodes,prelen]
            elif args.model != 'gwnet':
                preds,_,_ = engine.model(testx)
                preds = preds.transpose(1, 3)
            else:
                preds = engine.model(testx)
                preds = preds.transpose(1, 3)
            
        outputs.append(preds.squeeze())  #[batch_size, feature, pred_len]      

    yhat = torch.cat(outputs, dim=0)                                            
    yhat = yhat[:realy.size(0), ...]                                            
    print("Training finished")                                                  
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))             

    amae = []
    amape = []
    armse = []
    ar2 = []
    prediction = yhat
    for i in range(args.pred_len):# pred = prediction[:, :, :i+1]
        pred = scaler.inverse_transform(yhat[:, :, :i+1]) if args.scaler else yhat[:, :, :i+1]
        
        real = realy[:, :, :i + 1]# [batch_size, num_nodes, pred_len]
        
        if args.site == 'Allsite':
            metrics = util.metric(pred, real)
        else :
            metrics = util.metric(pred[:,vmindex_in_site,:], real[:,vmindex_in_site,:])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test SMAPE: {:.4f}, Test MSE: {:.4f}, Test R^2: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        ar2.append(metrics[3])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test SMAPE: {:.4f}, Test MSE: {:.4f}, Test R^2: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(ar2)))                      
    torch.save(engine.model.state_dict(), params_path+"/"+args.model+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid], 2))+".pth")
    prediction_path = params_path+"/"+args.model+"_prediction_results"
    ground_truth = realy.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
  
    np.savez_compressed(          
            os.path.normpath(prediction_path),
            prediction=prediction,
            ground_truth=ground_truth
    )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
