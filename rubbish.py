
        self.patch_len = 24
        self.step = 24
        print("patch_len:",self.patch_len,"step:",self.step)
        self.num_patch = 1 + (length - self.patch_len) // self.step
        tem_size = self.patch_len # should equal to x.dim3 in forward   

def generate_ts_token(self, x_inp):

        # print("before:",x_inp.shape)
        x_inp = x_inp.unfold(dimension=-1, size=self.patch_len, step=self.step)
        # print("unfold:",x_inp.shape)[b,1,n,np,p]
        x_inp = x_inp.squeeze(1)
        x_embed = self.feature_depthwise(x_inp)# b,n,np,p
        x_embed = self.feature_embedding(x_embed.transpose(2,3))# b,n,p,1 
       

        x_embed = x_embed.permute(0,3,1,2) # b,n,p,1 -->b,1,n,p
        # print("linear:",x_embed.shape)
        #x_embed = self.ts_embed_dropout(x_embed)
        # 和卷积压缩（压缩片、压缩原输入）做权重融合？
        return x_embed


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


def contrastive(self, x):
        anchor = x[:, :, self.vmindex[0], :]  # 假设index[0]是锚点的索引
        positive_samples = x[:, :, self.vmindex[1:], :]  # 正样本
        a = set(range(x.shape[2]))
        b = set(self.vmindex.tolist())
        c = list(a-b)
        d = torch.tensor(c, device=self.device)
        negative_indices = torch.tensor(list(set(range(x.shape[2])) - set(self.vmindex.tolist())), device=self.device)
        # 使用负样本的索引来获取负样本
        negative_samples = x[:, :, negative_indices, :] # 负样本

        # 计算对比损失（示例采用简单的欧氏距离，实际可根据需要使用复杂的对比损失函数）
        positive_loss = torch.mean((anchor - positive_samples) ** 2)
        negative_loss = torch.mean((anchor - negative_samples) ** 2)
        contrastive_loss = positive_loss - negative_loss  # 这里是简化版本，实际中可能需要更复杂的计算
        return contrastive_loss

        

        pre_gcnw = pretrained_dict['gcn.mlp.weight']
        self.model.inv_gcn.inv_mlp.weight = torch.nn.Parameter(pre_gcnw)
        if data_type == 'UP':
            self.alpha = 0.385
        else :
            self.alpha = 0.550
        self.loss = util.masked_mae
        self.loss2 = util.masked_mape

        loss1 = self.loss(predict, real, 0.0)
        loss2 = self.loss2(predict, real, 0.0)
        loss =self.alpha * loss1 + (1 - self.alpha) * loss2 

        self.first_conv = nn.ConvTranspose2d(in_channels=1, out_channels=num_nodes, kernel_size=(1, 1)) 
        self.inv_gcn = mgcn_depoly(1,num_nodes,len(supports))
        self.W = nn.Parameter(torch.tensor([1, 0, 0], dtype=torch.float))
        self.gate1 = nn.Linear(num_nodes, 3)
        self.gate2 = nn.Linear(num_nodes, 3)
        self.swish = Swish()
        self.mish = Mish()
        self.disagg_type = disagg_type
        ## plus plus
       
        y = assist_ten[:, :, 0, :].unsqueeze(1) 
        y = self.inv_gcn(y.transpose(2, 3)).transpose(1, 2).transpose(1,3)
        # *************
        if self.disagg_type == 1:
            w0, w1, w2 = F.softmax(self.W)
            output = w0 * x + w1 * torch.tanh(y) * x + w2 * torch.sigmoid(y) * x 
        elif self.disagg_type == 2:
            batch, length, feature, _ = x.size()

            # 由于最后一维是1，我们可以直接将x和y展平为[Batch * Length, Feature]
            x_flat = x.reshape(batch * length, feature)
            y_flat = y.reshape(batch * length, feature)
            # feature的压缩,改成3个通道
            # 计算门控信号
            gates = self.swish(self.gate1(x_flat) + self.gate2(y_flat))  # 输出维度为[Batch * Length, 3]
            gates = F.softmax(gates, dim=-1).view(batch, length, 3, 1)  # 转换回[Batch, Length, 3]并应用softmax

            # 使用门控信号控制每部分的贡献
            part1 = gates[:, :, 0, :].unsqueeze(2) * self.mish(x)
            part2 = gates[:, :, 1, :].unsqueeze(2) * torch.tanh(y) 
            part3 = gates[:, :, 2, :].unsqueeze(2) * torch.sigmoid(y) 
            # 融合这三部分
            output = part1 + part2 + part3
        return output