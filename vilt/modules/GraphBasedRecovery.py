import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import BertTextEncoder
from utils.transformers_encoder.transformer import TransformerEncoder
from utils.glow import Glow, ZeroConv2d, gaussian_log_p
import numpy as np
from utils.rcan import Group
from random import sample
import easydict

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real, missing_type):
        '''
        print("MSE内部：pred.shape=",pred.shape)
        print("MSE内部：real.shape=",real.shape)
        '''
        if pred.dim() == 2:
            # 添加一个新的维度，索引为-1，表示最后一个位置
            pred = pred.unsqueeze(-1)
        if real.dim() == 2:
            # 添加一个新的维度，索引为-1，表示最后一个位置
            real = real.unsqueeze(-1)
        if missing_type==1:
            pred_sliced = pred[:real.size(0), :, :] 
            diffs = torch.add(real, -pred_sliced)
        elif missing_type==0:
            real_sliced = real[:pred.size(0), :, :]   #！！！！！！！！逼不得已
            diffs = torch.add(real_sliced, -pred)
        #print("MSE里，missing_type=",missing_type)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class GraphBasedRecovery(nn.Module):
    def __init__(self, args):
        args=easydict.EasyDict(args)
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained='/data/zc/PVLM-main/bert-base-uncased')
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads # 目标特征维度，多头注意力头数 [32,8]
        self.orig_d_l, self.orig_d_v = args.feature_dims # 原始文本、图像特征维度
        self.d_l  = self.d_v = dst_feature_dims # 统一为目标特征维度
        self.num_heads = nheads # 多头注意力头数
        self.layers = args.nlevels # Transformer编码层数
        self.attn_dropout = args.attn_dropout # 注意力dropout参数
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout # ReLU dropout参数
        self.embed_dropout = args.embed_dropout # 嵌入dropout参数
        self.res_dropout = args.res_dropout # 残差连接dropout参数
        self.output_dropout = args.output_dropout # 输出dropout参数
        self.text_dropout = args.text_dropout # 文本dropout参数
        self.attn_mask = args.attn_mask # 注意力mask
        self.MSE = MSE()

        combined_dim = (self.d_l + self.d_v) # 联合特征维度
        #print("combined_dim=",combined_dim)
        output_dim = args.num_classes if args.train_mode == "classification" else 1 # 输出维度(分类或回归)

        # 可逆流模型
        self.flow_l = Glow(in_channel=self.d_l, n_flow=32, n_block=1, affine=True, conv_lu=False)
        self.flow_v = Glow(in_channel=self.d_v, n_flow=32, n_block=1, affine=True, conv_lu=False)

        #高斯分布先验参数,通过ZeroConv自适应学习
        self.prior_l = ZeroConv2d(self.d_l, self.d_l * 2)
        self.prior_v = ZeroConv2d(self.d_l, self.d_l * 2)
  

        # 语音、视频、音频的数据重构模块
        self.rec_l = nn.Sequential(
            nn.Conv1d(self.d_l, self.d_l*2, 1),
            Group(num_channels=self.d_l*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_l*2, self.d_l, 1)
        )

        self.rec_v = nn.Sequential(
            nn.Conv1d(self.d_v, self.d_v*2, 1),
            Group(num_channels=self.d_v*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_v*2, self.d_v, 1)
        )



        # 不同模态的特征拼接
        self.cat_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_v = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)

        # 1. Temporal convolutional layers 浅层卷积将不同模态投影到统一维度
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Crossmodal Attentions 交叉模态间的自注意力
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_v_with_l = self.get_network(self_type='vl')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.) 模态内的自注意力
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # weight for each modality 每个模态的权重
        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)

        # Projection layers 多模态融合的全连接层
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1): # 获取自注意力网络,使用不同的超参数 同模态使用不同的embed_dim、attn_dropout参数
        if self_type in ['l', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['v', 'lv']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    

    

    def forward(self, text, video, missing_type=None, single_missing_type=None): 
        '''
        # 构造损失函数
        def information_bottleneck_loss(logdet, log_p_sum):
            def entropy(logits):
                probs = torch.softmax(logits, dim=-1)
                return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # 添加一个很小的数值以防止log(0)的情况出现
            # 计算信息熵
            entropy_term = entropy(log_p_sum)
            # 计算条件熵
            conditional_entropy_term = entropy(logdet)
            # 计算信息瓶颈损失
            loss = entropy_term - conditional_entropy_term
            # 返回损失
            return loss
        '''
        mask = torch.tensor(single_missing_type) == 0  # 布尔掩码，形状是 [16]

        # 检查掩码的维度
        #print("mask shape =", mask.shape)  # 应该是 torch.Size([16])


        if self.use_bert:
            with torch.no_grad():
                text = self.text_model(text)
        #将文本、音频、视频转置为[batch, seq_len, feature_dim]格式
        #对文本进行dropout
        x_l = F.dropout(text, p=self.text_dropout, training=self.training)
        x_v = video
        '''
        print("x_l=",x_l.shape) # [16, 512, 768]
        print("x_v=",x_v.shape) # [16, 217, 768]
        
        # Project the textual/visual features 原始特征维度投影到目标维度
        print("self.orig_d_l=",self.orig_d_l) #768
        print("self.d_l=",self.d_l) #512
        print("self.orig_d_v=",self.orig_d_v) #768
        print("self.d_v=",self.d_v) #512
        '''
        
        with torch.no_grad():
            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        # 保存投影后的特征
        conv_feat_l, conv_feat_v = proj_x_l, proj_x_v
        #print("保存一开始投影后的特征，conv_feat_l.shape=",conv_feat_l.shape) # [512, 768, 8]
        #print("保存一开始投影后的特征，conv_feat_v.shape=",conv_feat_v.shape) # [217, 768, 8]
        #  normalizing flow for language 文字通过流模型编码到隐空间,获取logdet和latent表示
        _, logdet_l, z_outs_l = self.flow_l(proj_x_l.unsqueeze(-1)) #proj_x_l.unsqueeze(-1)    [512, 768, 8, 1]
        z_l = z_outs_l
        z_outs_l = z_outs_l[0]

        #  normalizing flow for vision
        _, logdet_v, z_outs_v = self.flow_v(proj_x_v.unsqueeze(-1))
        z_v = z_outs_v
        z_outs_v = z_outs_v[0]
        '''
        print("len(z_outs_v)=",len(z_outs_v))
        print("len(z_outs_l)=",len(z_outs_l))
        '''

        # 遍历每个样本,根据样本标签,从先验分布中采样,计算log p
        log_p_sum_l, log_p_sum_v = 0.0, 0.0
        for i in range(text.size(0)): 
            zero = torch.zeros_like(z_outs_l[i]).unsqueeze(0)
            mean_l, log_sd_l = self.prior_l(zero).chunk(2, 1)  # learnable mean and log_sd with zeroconv
            log_p_sum_l += gaussian_log_p(z_outs_l[i].unsqueeze(0), mean_l, log_sd_l).view(1, -1).sum(1)

        for i in range(video.size(0)): 
            zero = torch.zeros_like(z_outs_v[i]).unsqueeze(0)
            mean_v, log_sd_v = self.prior_v(zero).chunk(2, 1)  # learnable mean and log_sd with zeroconv
            log_p_sum_v += gaussian_log_p(z_outs_v[i].unsqueeze(0), mean_v, log_sd_v).view(1, -1).sum(1)


        # 遍历每个样本,根据样本标签,从先验分布中采样,计算log p

        #lambda1,lambda2,lambda3=0.4,0.4,0.2
        log_p_l = logdet_l.sum() + log_p_sum_l
        #log_p_l = lambda1 * logdet_l.sum() + lambda2 * log_p_sum_l - lambda3 * torch.exp(entropy_l)
        log_p_l = torch.max(torch.zeros_like(log_p_l),
                            (-log_p_l / (np.log(2) * proj_x_l.size(0) * proj_x_l.size(1) * proj_x_l.size(2)))).sum()
        log_p_v = logdet_v.sum() + log_p_sum_v
        #log_p_v = lambda1 * logdet_v.sum() + lambda2 * log_p_sum_v - lambda3 * torch.exp(entropy_v)
        log_p_v = torch.max(torch.zeros_like(log_p_v),
                            (-log_p_v / (np.log(2) * proj_x_v.size(0) * proj_x_v.size(1) * proj_x_v.size(2)))).sum()
        
 
        if missing_type == 0:  # has video
            proj_x_l = self.flow_l.reverse(z_v, reconstruct=True).squeeze(-1).detach()
            proj_x_l = self.rec_l(proj_x_l)
            '''
            print('重构后，proj_x_l=',proj_x_l.shape)
            print('重构后，conv_feat_l=',conv_feat_l.shape)
            '''
            rec_l=proj_x_l.squeeze(-1)
            rec_v=''
            loss_rec = self.MSE(rec_l[mask,:,:], conv_feat_l[mask,:,:].detach(), missing_type) #proj_x_l预测值 conv_feat_l 真实值
        elif missing_type==1: # has text
            proj_x_v = self.flow_v.reverse(z_l, reconstruct=True).squeeze(-1).detach()
            proj_x_v = self.rec_v(proj_x_v)
            rec_l=''
            rec_v=proj_x_v.squeeze(-1)
            loss_rec = self.MSE(rec_v[mask,:,:], conv_feat_v[mask,:,:].detach(), missing_type)
        elif missing_type == 2:  # all missing
            proj_x_l = self.flow_l.reverse(z_v, reconstruct=True).squeeze(-1).detach()
            proj_x_l = self.rec_l(proj_x_l)
            proj_x_v = self.flow_v.reverse(z_l, reconstruct=True).squeeze(-1).detach()
            proj_x_v = self.rec_v(proj_x_v)
            rec_l=proj_x_l.squeeze(-1)
            rec_v=proj_x_v.squeeze(-1)

            loss_rec = (self.MSE(rec_l, conv_feat_l.detach(), 0) + self.MSE(rec_v, conv_feat_v.detach(), 1))/2
        elif missing_type==3: # no missing
            loss_rec = torch.tensor(0)


        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        #print('重构又permute后，proj_x_v=',proj_x_v.shape) #mmimdb([764, 4, 32])
        #print('重构又permute后，proj_x_l=',proj_x_l.shape) #mmimdb([764, 4, 32])
        
        # V --> L
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        #print('h_l_with_vs=',h_l_with_vs.shape) #mmimdb([764, 4, 32])
        h_ls = torch.cat([h_l_with_vs, h_l_with_vs], dim=2)
        #print('trans_l_mem前，h_ls=',h_ls.shape) #mmimdb([764, 4, 32])
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        #print('trans_l_mem后，h_ls=',h_ls.shape) #mmimdb([764, 8, 32])
        last_h_l = last_hs = h_ls


        # L --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        #print('h_v_with_ls=',h_v_with_ls.shape) #mmimdb([764, 4, 32])

        h_vs = torch.cat([h_v_with_ls, h_v_with_ls], dim=2)
        #print('trans_v_mem前，h_vs=',h_vs.shape) #mmimdb([764, 4, 64])
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        #print('trans_v_mem后，h_vs=',h_vs.shape) #mmimdb([764, 4, 64])
        last_h_v = last_hs = h_vs
        #print('last_h_v=',last_h_v.shape) #mmimdb([764, 4, 64])

        #print('last_h_l.shape=',last_h_l.shape) #([28, 128, 32])  mt=0的时候第一维都是45 mt=1时第一维都是28 mt=2时上面是45下面是28
        #print('last_h_v.shape=',last_h_v.shape) #([28, 128, 32])

        if last_h_l.size(0) > last_h_v.size(0):
            last_h_l = last_h_l[:last_h_v.size(0)]
        elif last_h_v.size(0) > last_h_l.size(0):
            last_h_v = last_h_v[:last_h_l.size(0)]
        last_hs = torch.cat([last_h_l, last_h_v], dim=1)
        # A residual block
        #print("last_hs.shape=",last_hs.shape) #mmimdb([764, 8, 64])
        self.proj1(last_hs)
        F.relu(self.proj1(last_hs), inplace=True)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        #print('last_hs_proj=',last_hs_proj.shape) #mmimdb([764, 8, 64])
        # proj1层ReLU激活后dropout,再通过proj2层进行线性投影,得到last_hs_proj
        output = self.out_layer(last_hs_proj)
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@output.shape=",output.shape) #mmimdb([764, 4, 16]) 16为class_nums
        # 通过输出层计算最终输出output
        res = {
            #'ava_modal_idx': ava_modal_idx,# 可用模态索引
            'rec_v':rec_v,
            'rec_l':rec_l,
            'log_p_l': log_p_l,
            'log_p_v': log_p_v,
            'loss_rec': loss_rec,
            'M': output
        }
        return res
