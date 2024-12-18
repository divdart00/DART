'''
***********************************************************************
DART: Diversified and Accurate Long-Tail Recommendation

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: model.py
- classes of models (DART, DeepMF, FFN)

Version: 1.0
***********************************************************************
'''

import torch
import numpy as np
from torch import nn
import os
import sys
from utils import *

'''
feed forawrd network for DART

input:
    * hidden_units: dimension of hidden vectors
    * dropout_rate: ratio for drop out technique

* outputs: layer output
'''
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.GELU() # 원래는 relu
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

'''
Proposed Model (DART) based on SASRec

input:
* user_num: number of users
* item_num: number of items
* args: pre-defined arguments using argparse
'''
class DART(torch.nn.Module):
    # initialization of model
    def __init__(self, user_num, item_num, args):
        super(DART, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.temperature= args.temperature
        self.args=args

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.nce_fct = nn.CrossEntropyLoss(reduction="sum")
        self.multi_criterion = nn.MultiLabelSoftMarginLoss(reduction='mean')
        self.soft = nn.Softmax(dim=1)
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    # gets features
    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    # forward propagation of the model
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, aug1, aug2, aug3):        
        log_feats = self.log2feats(log_seqs) 

        aug1_log_feats=self.log2feats(aug1)
        aug2_log_feats=self.log2feats(aug2)
        aug3_log_feats=self.log2feats(aug3)   

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, aug1_log_feats, aug2_log_feats, aug3_log_feats

    '''
    calculate contrastive learning loss

    input:
    * aug1_seuqence_vectors, aug2_seuqence_vectors, aug3_seuqence_vectors: synthetic sequences (head masking, strong replacement, weak replacement)
    * batch_size: size of batch

    returns:
    * contrastive learning loss
    '''
    def cl_loss_ce_soft(self, aug1_seuqence_vectors, aug2_seuqence_vectors, aug3_seuqence_vectors, batch_size):
        N = 3 * batch_size
        z = torch.cat((aug1_seuqence_vectors, aug2_seuqence_vectors, aug3_seuqence_vectors), dim=0).to(self.dev)
        cos = nn.CosineSimilarity(dim=2, eps=1e-8)
        sim = cos(z.unsqueeze(1), z.unsqueeze(0))
        sim = sim / self.temperature
        sim_i_j1= torch.diag(sim, batch_size).to(self.dev)
        sim_i_j2 = torch.diag(sim, 2*batch_size).to(self.dev)
        sim_j_i1 = torch.diag(sim, -batch_size).to(self.dev)
        sim_j_i2 = torch.diag(sim, -2*batch_size).to(self.dev)
        positive_samples = torch.cat((sim_i_j1, sim_j_i2, sim_i_j2, sim_j_i1), dim=0).reshape(2,N).T.to(self.dev)
        mask = torch.ones((N, N), dtype=bool).to(self.dev)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size*2 + i] = 0
            mask[batch_size*2 + i, i] = 0
        for i in range(2*batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        negative_samples = sim[mask]
        negative_samples=negative_samples.reshape(N,-1)
        labels = torch.zeros(N, N-1).to(self.dev)
        labels[:, :2] = 0.5 
        logits = torch.cat((positive_samples, negative_samples), dim=1).to(self.dev)
        activated=self.soft(logits)
        loss=torch.mean(-torch.sum(labels*torch.log(activated), dim=1))
        return loss

    # calculate full loss
    def cal_loss_3(self,pos_logits,pos_labels,neg_logits,neg_labels,indices, aug1_log_feats, aug2_log_feats, aug3_log_feats, args, epoch):
        # recommendation loss
        rec_loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        rec_loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])

        #contrastive loss
        cl_loss=self.cl_loss_ce_soft(aug1_log_feats[:, -1, :], aug2_log_feats[:,-1,:],aug3_log_feats[:,-1,:], args.batch_size)
        
        return rec_loss + args.max_cl_weight*cl_loss

    # predicts the recommendation score between all users and given item_indices when sequence history(log_seqs) is provided
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(torch.LongTensor(log_seqs).to(self.dev)) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] 

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) 

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits 

'''
Neural Matrix Factorization model

input:
* num_users: number of users
* num_items: number of items
* args: pre-defined arguments using argparse
'''
class DeepMF(torch.nn.Module):
    def __init__(self, num_users, num_items, args):
        super(DeepMF, self).__init__()
        self.user_emb = torch.nn.Embedding(num_users, args.factor_num)
        self.item_emb = torch.nn.Embedding(num_items, args.factor_num)
        self.activation = torch.nn.ReLU()
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(torch.nn.ReLU())
        self.affine_output = torch.nn.Linear(in_features=args.layers[-1], out_features=1)

    def forward(self, user_idx, item_idx):
        out = torch.cat((self.user_emb(user_idx), self.item_emb(item_idx)), dim=1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            out = self.activation(self.fc_layers[idx](out))

        out = self.activation(self.affine_output(out))
        return out    