import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import knn, batch_choice
import open3d as o3d
import scipy.io as io

# def knn(x, k):
#     inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     distance = -xx - inner - xx.transpose(2, 1).contiguous()
#
#     idx = distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Propagate(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Propagate, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)

    def forward(self, x, idx):
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)  # [B, C, N, k]
        x = nn_feat - x.unsqueeze(-1)  # [B, c, N, k] - [B, c, N, 1] = [B, c, N, k]
        x = self.conv2d(x)  # [B, emb_dims, N, k]
        x = x.max(-1)[0]  # [B, C, N]
        x = self.conv1d(x)
        return x


class GNN(nn.Module):
    def __init__(self, emb_dims=64):
        super(GNN, self).__init__()
        self.propogate1 = Propagate(3, 64)  # Conv2DBNReLU(3,64)->Conv2DBNReLU(64,64)->Conv1DBNReLU(64,64)
        self.propogate2 = Propagate(64, 64)
        self.propogate3 = Propagate(64, 64)
        self.propogate4 = Propagate(64, 64)
        self.propogate5 = Propagate(64, emb_dims)

    def forward(self, x):
        # [B, 3, N]
        nn_idx = knn(x, k=12)  # [B, N, k], 最近邻索引

        x = self.propogate1(x, nn_idx)
        x = self.propogate2(x, nn_idx)
        x = self.propogate3(x, nn_idx)
        x = self.propogate4(x, nn_idx)
        x = self.propogate5(x, nn_idx)  # [B, emb_dims, N]

        return x

class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, src_corr, weights):
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered * weights.unsqueeze(1), src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, (weights.unsqueeze(1) * src).sum(dim=2, keepdim=True)) + (weights.unsqueeze(1) * src_corr).sum(dim=2, keepdim=True)
        return R, t.view(src.size(0), 3)
        # return R, t.view(batch_size, 3)

class PointNet(nn.Module):
    def __init__(self, in_dim, gn, out_dims, cls=False):
        super(PointNet, self).__init__()
        self.cls = cls
        l = len(out_dims)
        self.backbone = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.backbone.add_module(f'pointnet_conv_{i}',
                                     nn.Conv1d(in_dim, out_dim, 1, 1, 0))
            if gn:
                self.backbone.add_module(f'pointnet_gn_{i}',
                                         nn.GroupNorm(8, out_dim))
            if self.cls and i != l - 1:
                self.backbone.add_module(f'pointnet_relu_{i}',
                                         nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x, pooling=True):
        f = self.backbone(x)
        if not pooling:
            return f
        g, _ = torch.max(f, dim=2)
        return f, g

class HFECNet(nn.Module):
    def __init__(self, emb_nn, args):
        super(HFECNet, self).__init__()
        self.emb_dims = args.emb_dims  # 64
        self.num_iter = args.num_iter  # 4
        self.emb_nn = emb_nn  # GNN,[B, 64, N]
        self.K_test = args.K_test
        self.significance_fc = Conv1DBlock((self.emb_dims, 64, 32, 1), 1)
        self.head = SVDHead(args=args)
        self.sim_mat_Fconv = nn.ModuleList(
            [Conv2DBlock((self.emb_dims + 1, 32, 32, 32, 1), 1) for _ in range(self.num_iter)])
        self.sim_mat_Sconv = nn.ModuleList(
            [Conv2DBlock((7, 32, 32, 32, 1), 1) for _ in range(self.num_iter)])

        self.weight_fc = nn.ModuleList([Conv1DBlock((32,32,16,1), 1) for _ in range(self.num_iter)])
        self.preweight = nn.ModuleList([Conv2DBlock((2,16,32), 1) for _ in range(self.num_iter)])
        self.encoder = PointNet(in_dim=3,
                                gn=False,
                                out_dims=[64, 64, 128,256, 512])
        self.decoder_ol = PointNet(in_dim=2048,
                                   gn=False,
                                   out_dims=[512, 256, 128, 64,3],
                                   cls=True)
        self.encoder1 = PointNet(in_dim=64,
                                   gn=False,
                                   out_dims=[64, 128,256, 512])
        self.decoder_ol1 = PointNet(in_dim=2048,
                                   gn=False,
                                   out_dims=[512, 256, 128, 64],
                                   cls=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, R_gt=None, t_gt=None):
        """
        :param src: [B, 3, 768]
        :param tgt: [B, 3, 768]
        :param R_gt: [B, 3, 3]
        :param t_gt: [B, 3]
        :return:
        """
        ##### only pass ground truth while training #####
        if not (self.training or (R_gt is None and t_gt is None)):
            raise Exception('Passing ground truth while testing')

        ##### getting ground truth correspondences #####
        if self.training:
            src_gt = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1)
            dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)  # [B, 3, 768, 768]
            min_dist, min_idx = (dist ** 2).sum(1).min(-1) # [B, npoint], [B, npoint]
            min_dist = torch.sqrt(min_dist)
            min_idx = min_idx.cpu().numpy() # drop to cpu for numpy
            match_labels = (min_dist < 0.05).float()  # [B, 768]
            indicator = match_labels.cpu().numpy()
            indicator += 1e-5
            pos_probs = indicator / indicator.sum(-1, keepdims=True)  # [B, N]
            indicator = 1 + 1e-5 * 2 - indicator
            neg_probs = indicator / indicator.sum(-1, keepdims=True)  # get the proability whether the point in src has a correspondences in tgt

        ##### Keypoints' Feature Extraction #####
        #print("src",src.shape)16,3,768
        #print(tgt.shape)16,3,768
        tgt_embedding = self.emb_nn(tgt)
        src_embedding = self.emb_nn(src)  # [B, 64, N]
        x1 = src_embedding.contiguous()
        f_x1, g_x1 = self.encoder1(src_embedding)
        y1 = tgt_embedding.contiguous()
        f_y1, g_y1 = self.encoder1(y1)
        concat = torch.cat((g_x1, g_y1), dim=1)
        g_x_expand1 = torch.unsqueeze(g_x1, dim=-1).expand_as(f_x1)
        g_y_expand1 = torch.unsqueeze(g_y1, dim=-1).expand_as(f_y1)
        f_x_ensemble1 = torch.cat([f_x1, g_x_expand1, g_y_expand1,
                                  g_x_expand1 - g_y_expand1], dim=1)
        f_y_ensemble1 = torch.cat([f_y1, g_y_expand1, g_x_expand1,
                                  g_y_expand1 - g_x_expand1], dim=1)#n*4c

        src_embedding = self.decoder_ol1(f_x_ensemble1, pooling=False)#n*64
        #print(x_ol.shape)
        tgt_embedding = self.decoder_ol1(f_y_ensemble1, pooling=False)
        #print(src_embedding.shape)#16,64,768
        src_sig_score = self.significance_fc(src_embedding).squeeze(1)  # [B, N]
        #print(src_sig_score.shape)#16,768
        tgt_sig_score = self.significance_fc(tgt_embedding).squeeze(1)

        num_point_preserved = src.size(-1) // 6

        if not self.training:
            num_point_preserved = self.K_test

        if self.training:
            candidates = np.tile(np.arange(src.size(-1)), (src.size(0), 1))
            pos_idx = batch_choice(candidates, num_point_preserved//2, p=pos_probs)
            neg_idx = batch_choice(candidates, num_point_preserved-num_point_preserved//2, p=neg_probs)
            src_idx = np.concatenate([pos_idx, neg_idx], 1)
            tgt_idx = min_idx[np.arange(len(src))[:, np.newaxis], src_idx]
        else:
            src_idx = src_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            src_idx = src_idx.cpu().numpy()
            tgt_idx = tgt_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            tgt_idx = tgt_idx.cpu().numpy()
        batch_idx = np.arange(src.size(0))[:, np.newaxis]
        if self.training:
            match_labels = match_labels[batch_idx, src_idx]
        src = src[batch_idx, :, src_idx].transpose(1, 2)  # [B, 3, N]
        #print("src",src.shape)16,3,256
        src_embedding = src_embedding[batch_idx, :, src_idx].transpose(1, 2)  # [B, 3, C]
        #print("src2",src_embedding.shape)16,64,256
        src_sig_score = src_sig_score[batch_idx, src_idx]

        tgt = tgt[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_embedding = tgt_embedding[batch_idx, :, tgt_idx].transpose(1, 2)      
        tgt_sig_score = tgt_sig_score[batch_idx, tgt_idx]


        ##### transformation initialize #####
        R = torch.eye(3).unsqueeze(0).expand(src.size(0), -1, -1).cuda().float()
        t = torch.zeros(src.size(0), 3).cuda().float()
        loss = 0.

        for i in range(self.num_iter):
            batch_size, num_dims, num_points = src_embedding.size()
            _src_emb = src_embedding.unsqueeze(-1).repeat(1, 1, 1, num_points)
            
            #print(_src_emb.shape)16,64,256,256
            _tgt_emb = tgt_embedding.unsqueeze(-2).repeat(1, 1, num_points, 1)  # [B, C, N, N]
            
            #print(_tgt_emb.shape)16,64,256,256

            #### Feature Matching Matrix Computation ####
            diff_f = _tgt_emb - _src_emb
            #print(diff_f.shape)16,64,256,256
            dist_f = torch.sqrt((diff_f ** 2).sum(1, keepdim=True))
            diff_f = diff_f / (dist_f + 1e-8)
            similarity_matrix_F = torch.cat([dist_f, diff_f], 1)# [B, 65, N, N]
            similarity_matrix_F = self.sim_mat_Fconv[i](similarity_matrix_F)  # [B, 1, N, N]

            ##### Coordinate Matching Matrix Computation #####
            diff = src.unsqueeze(-1) - tgt.unsqueeze(-2)
            
            #print(diff.shape)16,3,256,256
            dist = (diff ** 2).sum(1, keepdim=True)
            dist = torch.sqrt(dist)
            diff = diff / (dist + 1e-8)
            similarity_matrix_S = torch.cat([dist, diff, src.unsqueeze(-1).repeat(1, 1, 1, tgt.size(2))], 1)  # [B, 7, N, N]
            similarity_matrix_S = self.sim_mat_Sconv[i](similarity_matrix_S)  # [B, 1, N, N]

            ##### Final Matching Matrix Computation #####
            similarity_matrix = similarity_matrix_F + similarity_matrix_S#sum

            ##### Correspondences Credibility Computation #####
            preweight = torch.cat([similarity_matrix_F, similarity_matrix_S], 1)#concat
            preweight = self.preweight[i](preweight)
            #print(preweight.shape)[16, 32, 256, 256])

            #preweight = self.softmax(preweight)
            weights = preweight.max(-1)[0]
            #print(weights.shape)([16, 32, 256])
            preweight1 = self.softmax(weights)
            weights=torch.mul(weights,preweight1)
            weights = self.weight_fc[i](weights).squeeze(1)  # [B, N]
            #print(weights.shape)([16, 256])
            

            ##### Obtain  Final Matching Matrix #####
            similarity_matrix = similarity_matrix.squeeze(1)
            similarity_matrix = similarity_matrix.clamp(min=-20, max=20)  # [B, N, N] in -20 ~ 20
            ##### similarity matrix convolution to get similarities #####

            ###############################################      Loss     ################################################################
            ##### keypoints selection loss #####
            if self.training and i==0:
                src_neg_ent = torch.softmax(similarity_matrix.squeeze(1), dim=-1)
                src_neg_ent = (src_neg_ent * torch.log(src_neg_ent)).sum(-1)
                tgt_neg_ent = torch.softmax(similarity_matrix.squeeze(1), dim=-2)
                tgt_neg_ent = (tgt_neg_ent * torch.log(tgt_neg_ent)).sum(-2)
                loss = loss + (F.mse_loss(src_sig_score, src_neg_ent.detach()) + F.mse_loss(tgt_sig_score, tgt_neg_ent.detach()))

            ###### correspondence matching loss #####
            if self.training:
                temp = torch.softmax(similarity_matrix, dim=-1)  # [B, N. N]
                temp = temp[:, np.arange(temp.size(-2)), np.arange(temp.size(-1))]
                temp = - torch.log(temp)
                match_loss = (temp * match_labels).sum() / match_labels.sum()
                loss = loss +  match_loss

            ##### finding correspondences #####
            corr_idx = similarity_matrix.max(-1)[1]
            src_corr = tgt[np.arange(tgt.size(0))[:, np.newaxis], :, corr_idx].transpose(1, 2)

            ##### correspondences credibility computation loss #####
            if self.training:
                weight_labels = (corr_idx == torch.arange(corr_idx.size(1)).cuda().unsqueeze(0)).float()
                weight_loss = F.binary_cross_entropy_with_logits(weights, weight_labels)
                loss = loss + weight_loss

            ##### Unreliable correspondence elimination #####
            weights = torch.sigmoid(weights)  # [B,N]
            weights = weights * (weights >= weights.median(-1, keepdim=True)[0]).float()
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)

            ##### get R and t #####
            rotation_ab, translation_ab = self.head(src, src_corr, weights)
            rotation_ab = rotation_ab.detach() # prevent backprop through svd
            translation_ab = translation_ab.detach() # prevent backprop through svd
            src = torch.matmul(rotation_ab, src) + translation_ab.unsqueeze(-1)
            R = torch.matmul(rotation_ab, R)
            t = torch.matmul(rotation_ab, t.unsqueeze(-1)).squeeze() + translation_ab

        return R, t, loss
