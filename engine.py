import torch
import torch.optim as optim
from model import *
import util
from adabelief_pytorch import AdaBelief

class trainer_global():#Global model, used in train_whole
    def __init__(self, scaler, in_dim, seq_length,pred_len , num_nodes, nhid, dropout, lrate, wdecay, device,
                 supports,l_supports, decay, site_num_nodes, rep_idx, gat=False, addaptadj=True, data_type='CPU',contrastive = 'RNC'):
        self.model = gwnet_gl(device, num_nodes, site_num_nodes, rep_idx, dropout, supports=supports, length=seq_length, in_dim=in_dim, out_dim=pred_len,
                              residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                              end_channels=nhid * 16, gat=gat, addaptadj=addaptadj, l_supports=l_supports, contrastive = contrastive)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        if data_type == 'CPU':
            self.loss = util.custom_loss_cpu_site
        else:
            self.loss = util.custom_loss_bw_site
        self.scaler = scaler
        self.clip = 5
        self.idx = rep_idx
        self.device = device

    def train(self, input, input_site, real_val, real_site):
        self.model.train()              
        self.optimizer.zero_grad()      
        output0, contrastive_loss, output_re = self.model(input, input_site, real_site)  # output = [batch_size, len, num_nodes, 1]
        output0 = output0.transpose(1, 3)  # site   
        output_re = output_re.transpose(1, 3).cpu().detach().to(self.device)  # [batch_size, 12, num_nodes, 1]

        real = torch.unsqueeze(real_val, dim=1)                                        
        real[:, :, self.idx, :] = output_re[:, :, 0, :]               
        real_site = torch.unsqueeze(real_site, dim=1)

        predict0 = self.scaler.inverse_transform(output0) if self.scaler is not None else output0 
        
        # real [b,1,n,len] 
        loss = self.loss(predict0, real, 0.0)                                          
        loss = loss + contrastive_loss                   
        loss.backward()                                                                 
        if self.clip is not None:                                                       
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()                                                           
        mae = util.masked_mae(predict0, real, 0.0).item()                               
        mape = util.masked_mape(predict0, real, 0.0).item()
        rmse = util.masked_rmse(predict0, real, 0.0).item()
        r2 = util.masked_r_squared(predict0, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, input_site, real_val, real_site):
        self.model.eval()                                           
        # input = nn.functional.pad(input,(1,0,0,0))
        output0, contrastive_loss, output_re = self.model(input, input_site, real_site)
        output0 = output0.transpose(1, 3)
        
        output_re = output_re.transpose(1, 3).cpu().detach().to(self.device)    
        # output = [batch_size,12,num_nodes,1]                                  
                                                                                
        real = torch.unsqueeze(real_val, dim=1)
        real[:, :, self.idx, :] = output_re[:, :, 0, :]     
        real_site = torch.unsqueeze(real_site, dim=1)

        predict0 = self.scaler.inverse_transform(output0) if self.scaler is not None else output0
        
        
        loss = self.loss(predict0, real, 0.0)
        loss = loss + contrastive_loss
        mae = util.masked_mae(predict0, real, 0.0).item()
        mape = util.masked_mape(predict0, real, 0.0).item()
        rmse = util.masked_rmse(predict0, real, 0.0).item()
        r2 = util.masked_r_squared(predict0, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2 


class trainer0():
    def __init__(self, scaler, in_dim, seq_length, pred_len, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, decay, data, site,  gat=False, addaptadj=True, data_type='CPU',
                 disagg_type=2):

        pretrained_dict = torch.load("./garage/gwnet/site2global/"
                                      + data + '/' + site + '/' + data_type + '/' + "best.pth",
                                      map_location=device)
        pre_w = pretrained_dict['first_conv.weight']
        pre_gcnw = pretrained_dict['gcn.mlp.weight']
        pre_b = torch.zeros(num_nodes)
        # cur_w = torch.mm(pre_w, torch.cholesky_inverse(torch.mm(pre_w.T, pre_w))).T
        self.model = gwnet_plus(device, num_nodes, dropout, supports=supports, in_dim=in_dim, out_dim=pred_len,length=seq_length,
                                residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                                end_channels=nhid * 16, gat=gat, addaptadj=addaptadj, disagg_type=disagg_type)
        self.model.first_conv.weight = torch.nn.Parameter(pre_w)
        self.model.inv_gcn.inv_mlp.weight = torch.nn.Parameter(pre_gcnw)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)

        if data_type == 'UP':
            self.alpha = 0.250
        else :
            self.alpha = 0.550
        print(self.alpha)
        self.loss = util.masked_mae
        self.loss2 = util.masked_mape
        self.scaler = scaler                           
        self.clip = 5

    def train(self, input, real_val, assist_ten):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, assist_ten)  # output = [batch_size, len , num_nodes, 1]
        output = output.transpose(1, 3)
        real = real_val
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output

        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2 
        # loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val, assist_ten):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input, assist_ten)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = real_val

        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2 
        # loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2


class trainer1():
    def __init__(self, scaler, in_dim, seq_length, pred_len, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, decay, gat=False, addaptadj=True, data_type='CPU', region='site'):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16, gat=gat, addaptadj=addaptadj)
        
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
       
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input) 
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val):
        self.model.eval()
       
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2
    
    
class trainer2():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = ASTGCN_Recent(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val, dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2    
    
class trainer3():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = GRCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
       
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2     
    
class trainer4():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = Gated_STGCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2  
    

    
class trainer5():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = H_GCN_wh(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2      

class trainer6():
    def __init__(self, in_dim,in_dim_cluster, seq_length, num_nodes, cluster_nodes, nhid , dropout, lrate, wdecay, device, supports,supports_cluster,transmit,decay):
        self.model = H_GCN_wdf(device, num_nodes,cluster_nodes, dropout, supports=supports, supports_cluster=supports_cluster,
                           in_dim=in_dim,in_dim_cluster=in_dim_cluster, out_dim=seq_length, transmit=transmit,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5
        self.supports=supports
        self.num_nodes=num_nodes

    def train(self, input, input_cluster, real_val,real_val_cluster):
        self.model.train()
        self.optimizer.zero_grad()
        output,output_cluster,tran2 = self.model(input,input_cluster)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        
        loss = self.loss(predict, real,0.0)
        (loss).backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, input_cluster, real_val,real_val_cluster):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input,input_cluster)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2
    
    
class trainer7():
    def __init__(self, in_dim,in_dim_cluster, seq_length, num_nodes, cluster_nodes, nhid , dropout, lrate, wdecay, device, supports,supports_cluster,transmit,decay):
        self.model = H_GCN(device, num_nodes,cluster_nodes, dropout, supports=supports, supports_cluster=supports_cluster,
                           in_dim=in_dim,in_dim_cluster=in_dim_cluster, out_dim=seq_length, transmit=transmit,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5
        self.supports=supports
        self.num_nodes=num_nodes

    def train(self, input, input_cluster, real_val,real_val_cluster):
        self.model.train()
        self.optimizer.zero_grad()
        output,output_cluster,tran2 = self.model(input,input_cluster)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        
        loss = self.loss(predict, real,0.0)
        (loss).backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, input_cluster, real_val,real_val_cluster):
        self.model.eval()
        output,_,_ = self.model(input,input_cluster)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2
    
class trainer8():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = OGCRNN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2
    
class trainer9():
    def __init__(self, in_dim, seq_length,pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = OTSGGCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2
    
    
class trainer10():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = LSTM(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
       
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,rmse,r2

    def eval(self, input, real_val):
        self.model.eval()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,rmse,r2 
    
    
class trainer11():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = GRU(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output,_,_ = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)  # real in shape [batch_size, 1, num_nodes, pre_len]
        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        mse = util.masked_mse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,mse,r2

    def eval(self, input, real_val):
        self.model.eval()
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        mse = util.masked_mse(predict, real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, mse, r2

class trainer12():
    def __init__(self, num_nodes, seq_length, pred_len, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = Informer(num_nodes,num_nodes,num_nodes, out_len=pred_len, dropout=dropout, device=device)

        self.model.to(device)
        self.device=device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        self.pred_len=pred_len
        self.clip = 5
    """
    batch_x: corresponds to the shape of the input sequence data (batch_size, seq_len, num_features), where batch_size is the number of samples in the batch.

    batch_y: corresponds to the shape of the target sequence data (batch_size, label_len + pred_len, num_features).

    batch_x_mark: contains the timestamp information of the input sequence, shape is (batch_size, seq_len, num_timestamp_features).

    batch_y_mark: contains timestamp information of the target sequence, shape is (batch_size, label_len + pred_len, num_timestamp_features).
    """
    #btrainx, x_mark, dec_inp, y_mark
    def train(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.train()
        self.optimizer.zero_grad()
        
        
        output = self.model(input, input_mark, dec_inp, real_val_mark)#x_enc, x_mark_enc, x_dec, x_mark_dec,
        #output = [batch_size,pre_len,num_nodes]
        #real = torch.unsqueeze(real_val, dim=1)  # real in shape [batch_size, 1, num_nodes, pre_len]
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val, 0.0)# real val= [batch_size,pre_len,num_nodes]
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(),mae,mape,mse,r2

    def eval(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.eval()
        output = self.model(input, input_mark, dec_inp, real_val_mark)
        
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val,0.0)
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict, real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(), mae, mape, mse, r2
    
class trainer13():
    def __init__(self, num_nodes, seq_length, pred_len, label_len, dropout, lrate, wdecay, device, supports,decay):
        self.model = Autoformer(num_nodes,num_nodes,num_nodes, out_len=pred_len, label_len=label_len, dropout=dropout, device=device)
        self.model.to(device)
        self.device=device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        self.pred_len=pred_len
        self.clip = 5
    #btrainx, x_mark, dec_inp, y_mark
    def train(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, input_mark, dec_inp, real_val_mark)#x_enc, x_mark_enc, x_dec, x_mark_dec,
       
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(),mae,mape,mse,r2

    def eval(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.eval()
        output = self.model(input, input_mark, dec_inp, real_val_mark)
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val,0.0)
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict, real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(), mae, mape, mse, r2

class trainer14():
    def __init__(self, iterations, seq_length, pred_len, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = N_BEATS(seq_length, pred_len)
        self.model.to(device)
        self.learning_rate = lrate #
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5
        self.lr_decay_step = iterations // 3
        if self.lr_decay_step == 0:
            self.lr_decay_step = 1

    def train(self, input, real_val, iter):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1, 3)
        predict = output

        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate * 0.5 ** (iter // self.lr_decay_step)

        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(),mae,mape,mse,r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        
        predict = output
        loss = self.loss(predict, real_val,0.0)
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict, real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(), mae, mape, mse, r2
    
class trainer15():
    def __init__(self, num_nodes, seq_length, pred_len, label_len, dropout, lrate, wdecay, device, supports,decay):
        self.model = TimesNet(num_nodes,num_nodes,num_nodes, seq_len=seq_length, pred_len=pred_len, label_len=label_len, dropout=dropout)
        self.model.to(device)
        self.device=device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        self.pred_len=pred_len
        self.clip = 5
    #btrainx, x_mark, dec_inp, y_mark
    def train(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, input_mark, dec_inp, real_val_mark)#x_enc, x_mark_enc, x_dec, x_mark_dec,
       
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(),mae,mape,mse,r2

    def eval(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.eval()
        output = self.model(input, input_mark, dec_inp, real_val_mark)
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val,0.0)
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict, real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(), mae, mape, mse, r2

class trainer16():
    def __init__(self, in_dim, seq_length, pre_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = DCRNN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, seq_len=seq_length, horizon=pre_len)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input[:, 0, :, :],real_val)
        
        real=real_val
        predict = output
        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input[:, 0, :, :],real_val)
        output = torch.unsqueeze(output,dim=1)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        real = real.transpose(1,3)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2
    
class trainer17():
    def __init__(self, seq_length, pred_len, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = NHITS(seq_length, pred_len)
        self.model.to(device)
        self.learning_rate = lrate #
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        real_val = torch.unsqueeze(real_val, dim=1)  # real in shape [batch_size, 1, num_nodes, pre_len]
        predict = output
        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(),mae,mape,mse,r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real_val,0.0)
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict, real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(), mae, mape, mse, r2
class trainer18():
    def __init__(self, batch_size, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = DeepAR(seq_length, pred_len, num_nodes, batch_size=batch_size, lstm_hidden_dim=nhid, 
                            lstm_dropout=dropout, device=device)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,rmse,r2

    def eval(self, input, real_val):
        self.model.eval()
        
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,rmse,r2 
######################################################################################################

class trainer_lyn_lstm():

    def __init__(self, scaler,  in_dim, seq_length, pred_len, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, decay, data, site,  gat=False, addaptadj=True, data_type='CPU',
                 disagg_type=2):

        pretrained_dict = torch.load("./garage/gwnet/site2global/"
                                      + data + '/' + site + '/' + data_type + '/' + "best.pth",
                                     map_location=device)           #loaded with pre-training weights.
        pre_w = pretrained_dict['first_conv.weight']                
        pre_gcnw = pretrained_dict['gcn.mlp.weight']
        self.model = LSTM_plus(device, num_nodes, dropout, supports=supports, in_dim=in_dim, out_dim=pred_len,length=seq_length,
                               residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                               end_channels=nhid * 16, disagg_type=disagg_type)
        self.model.first_conv.weight = torch.nn.Parameter(pre_w)
        self.model.inv_gcn.inv_mlp.weight = torch.nn.Parameter(pre_gcnw)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        if data_type == 'UP':
            self.alpha = 0.980
        else :
            self.alpha = 0.550
        self.loss = util.masked_mae
        self.loss2 = util.masked_mape
        self.scaler = scaler                                        
        self.clip = 5

    def train(self, input, real_val, assist_ten):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, assist_ten)  
        output = output.transpose(1, 3)
        real = real_val
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2 
        #loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2 

    def eval(self, input, real_val, assist_ten):
        self.model.eval()
        output = self.model(input, assist_ten)
        output = output.transpose(1, 3)
        real = real_val

        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2 
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2 


class trainer_lyn_OTSGGCN():

    def __init__(self, scaler, in_dim, seq_length, pred_len, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, decay, data, site,  gat=False, addaptadj=True, data_type='CPU',
                 disagg_type=2,):

        pretrained_dict = torch.load("./garage/gwnet/site2global/"
                                      + data + '/' + site + '/' + data_type + '/' + "best.pth",
                                     map_location=device)

        pre_w = pretrained_dict['first_conv.weight']
        pre_gcnw = pretrained_dict['gcn.mlp.weight']
        self.model = OTSGGCN_plus(device, num_nodes, dropout, supports=supports, in_dim=in_dim, out_dim=pred_len,length=seq_length,
                                  residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,end_channels=nhid * 16,
                                  disagg_type=disagg_type,)
        self.model.first_conv.weight = torch.nn.Parameter(pre_w)
        self.model.inv_gcn.inv_mlp.weight = torch.nn.Parameter(pre_gcnw)

        
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
                        
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        if data_type == 'UP':
            self.alpha = 0.385
        else :
            self.alpha = 0.550
        self.loss = util.masked_mae
        self.loss2 = util.masked_mape

        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, assist_ten):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, assist_ten)  
        output = output.transpose(1, 3)
        real = real_val
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2 
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val, assist_ten):
        self.model.eval()
        output = self.model(input, assist_ten)
        output = output.transpose(1, 3)
        real = real_val

        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2 

        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

class trainer_lyn_STGCN():

    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, decay, data, site,  gat=False, addaptadj=True, data_type='CPU',
                 disagg_type = 2):

        pretrained_dict = torch.load("./garage/gwnet/site2global/"
                                      + data + '/' + site + '/' + data_type + '/' + "best.pth",
                                     map_location=device)
        pre_w = pretrained_dict['first_conv.weight']
        pre_gcnw = pretrained_dict['gcn.mlp.weight']
        
        self.model = STGCN_plus(device, num_nodes, dropout, supports=supports, in_dim=in_dim, out_dim=seq_length,
                                residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                                end_channels=nhid * 16, disagg_type = disagg_type,)
        self.model.first_conv.weight = torch.nn.Parameter(pre_w)
        self.model.inv_gcn.inv_mlp.weight = torch.nn.Parameter(pre_gcnw)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        if data_type == 'UP':
            self.alpha = 0.385
        else :
            self.alpha = 0.480
        self.loss = util.masked_mae
        self.loss2 = util.masked_mape
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, assist_ten):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, assist_ten) 
        output = output.transpose(1, 3)
        real = real_val
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val, assist_ten):
        self.model.eval()
        output = self.model(input, assist_ten)
        output = output.transpose(1, 3)
        real = real_val
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2


class trainer_lyn_GRCN():

    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, decay, data, site,  gat=False, addaptadj=True, data_type='CPU',
                 disagg_type = 2):

        pretrained_dict = torch.load("./garage/gwnet/site2global/"
                                      + data + '/' + site + '/' + data_type + '/' + "best.pth",
                                     map_location=device)
        pre_w = pretrained_dict['first_conv.weight']
        pre_gcnw = pretrained_dict['gcn.mlp.weight']
        
        self.model = GRCN_plus(device, num_nodes, dropout, supports=supports, in_dim=in_dim, out_dim=seq_length,
                               residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                               end_channels=nhid * 16, disagg_type = disagg_type)
        self.model.first_conv.weight = torch.nn.Parameter(pre_w)
        self.model.inv_gcn.inv_mlp.weight = torch.nn.Parameter(pre_gcnw)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        if data_type == 'UP':
            self.alpha = 0.250
        else :
            self.alpha = 0.930
        self.loss = util.masked_mae
        self.loss2 = util.masked_mape
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, assist_ten):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, assist_ten) 
        output = output.transpose(1, 3)
        real = real_val
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val, assist_ten):
        self.model.eval()
        output = self.model(input, assist_ten)
        output = output.transpose(1, 3)
        real = real_val

        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

class trainer_lyn_ASTGCN():

    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, decay, data, site,  gat=False, addaptadj=True, data_type='CPU',
                 disagg_type = 2):

        pretrained_dict = torch.load("./garage/gwnet/site2global/"
                                      + data + '/' + site + '/' + data_type + '/' + "best.pth",
                                     map_location=device)
        pre_w = pretrained_dict['first_conv.weight']
        pre_gcnw = pretrained_dict['gcn.mlp.weight']
        
        self.model = ASTGCN_plus(device, num_nodes, dropout, supports=supports, in_dim=in_dim, out_dim=seq_length,
                               residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                               end_channels=nhid * 16, disagg_type = disagg_type)
        self.model.first_conv.weight = torch.nn.Parameter(pre_w)
        self.model.inv_gcn.inv_mlp.weight = torch.nn.Parameter(pre_gcnw)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        if data_type == 'UP':
            self.alpha = 0.250
        else :
            self.alpha = 0.550
        self.loss = util.masked_mae
        self.loss2 = util.masked_mape
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, assist_ten):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, assist_ten) 
        output = output.transpose(1, 3)
        real = real_val
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, real_val, assist_ten):
        self.model.eval()
        output = self.model(input, assist_ten)
        output = output.transpose(1, 3)
        real = real_val

        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

class trainer_lyn_Autoformer():

    def __init__(self, scaler, in_dim, seq_length, pred_len, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, decay, data, site,  gat=False, addaptadj=True, data_type='CPU',
                 disagg_type = 2):

        pretrained_dict = torch.load("./garage/gwnet/site2global/"
                                      + data + '/' + site + '/' + data_type + '/' + "best.pth",
                                     map_location=device)
        pre_w = pretrained_dict['first_conv.weight']
        pre_gcnw = pretrained_dict['gcn.mlp.weight']
        
        self.model = Autoformer_plus(num_nodes,num_nodes,num_nodes, seq_len=seq_length, pred_len=pred_len,
                                    label_len=0, dropout=dropout,supports = supports,disagg_type = disagg_type)
        self.model.first_conv.weight = torch.nn.Parameter(pre_w)
        self.model.inv_gcn.inv_mlp.weight = torch.nn.Parameter(pre_gcnw)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        if data_type == 'UP':
            self.alpha = 0.150
        else :
            self.alpha = 0.700
        self.loss = util.masked_mae
        self.loss2 = util.masked_mape
        self.scaler = scaler
        self.clip = 5

    def train(self, input, input_mark, real_val, dec_inp,real_val_mark, assist_ten):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, input_mark, dec_inp, real_val_mark, assist_ten)  # output = [batch_size, 12, num_nodes, 1]
        
        output = output.transpose(1, 3)# b 1 n ,len
        real = torch.unsqueeze(real_val, dim=3).transpose(1, 3)
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        #loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, input_mark, real_val, dec_inp,real_val_mark, assist_ten):
        self.model.eval()
        output = self.model(input, input_mark, dec_inp, real_val_mark, assist_ten)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=3).transpose(1, 3)

        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

class trainer_lyn_TimesNet():

    def __init__(self, scaler, in_dim, seq_length, pred_len, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, decay, data, site,  gat=False, addaptadj=True, data_type='CPU',
                 disagg_type = 2):

        pretrained_dict = torch.load("./garage/gwnet/site2global/"
                                      + data + '/' + site + '/' + data_type + '/' + "best.pth",
                                     map_location=device)
        pre_w = pretrained_dict['first_conv.weight']
        pre_gcnw = pretrained_dict['gcn.mlp.weight']
        
        self.model = TimesNet_plus(num_nodes,num_nodes,num_nodes, seq_len=seq_length, pred_len=pred_len,
                                    label_len=0, dropout=dropout,supports = supports,disagg_type = disagg_type)
        self.model.first_conv.weight = torch.nn.Parameter(pre_w)
        self.model.inv_gcn.inv_mlp.weight = torch.nn.Parameter(pre_gcnw)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        if data_type == 'UP':
            self.alpha = 0.150
        else :
            self.alpha = 0.700
        self.loss = util.masked_mae
        self.loss2 = util.masked_mape
        self.scaler = scaler
        self.clip = 5

    def train(self, input, input_mark, real_val, dec_inp,real_val_mark, assist_ten):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, input_mark, dec_inp, real_val_mark, assist_ten) 
        
        output = output.transpose(1, 3)# b 1 n ,len
        real = torch.unsqueeze(real_val, dim=3).transpose(1, 3)
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2

    def eval(self, input, input_mark, real_val, dec_inp,real_val_mark, assist_ten):
        self.model.eval()
        output = self.model(input, input_mark, dec_inp, real_val_mark, assist_ten)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=3).transpose(1, 3)

        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, r2