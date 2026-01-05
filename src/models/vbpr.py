# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""

import torch
import torch.nn as nn
from ..utils.recommender import GeneralRecommender
from ..utils.compute_loss import BPRLoss, EmbLoss
import torch.nn.functional as F


class VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        
        # Store normalized features (no computation graph)
        if self.v_feat is not None and self.t_feat is not None:
            self.text_linear = nn.Linear(self.t_feat.shape[1], self.i_embedding_size)
            self.vison_linear = nn.Linear(self.v_feat.shape[1], self.i_embedding_size)
            self.fusion_layer = nn.Linear(self.i_embedding_size * 2, self.i_embedding_size)
        elif self.v_feat is not None:
            self.text_linear = None
            self.vison_linear = nn.Linear(self.v_feat.shape[1], self.i_embedding_size)
        else:
            self.text_linear = nn.Linear(self.t_feat.shape[1], self.i_embedding_size)
            self.vison_linear = None

        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, dropout=0.0):
        if self.text_linear is not None and self.vison_linear is not None:
            combined_features = torch.cat((self.text_linear(self.t_feat), self.vison_linear(self.v_feat)), dim=-1)
            item_raw_features = self.fusion_layer(combined_features)
        elif self.vison_linear is not None:
            item_raw_features = self.vison_linear(self.v_feat)
        else:
            item_raw_features = self.text_linear(self.t_feat)
            
        item_embeddings = torch.cat((self.i_embedding, item_raw_features), -1)
        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
        return user_e, item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]
        
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score