import pickle
import torch.nn.functional as F
import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import math
import numpy as np
import scipy.sparse as sp
import numpy as np
import scipy.sparse as sp

import numpy as np
import torch
import torch.nn as nn


class HPT(SequentialRecommender):
    def __init__(self, config, dataset):
        super(HPT, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.dataset_name = config["dataset"]
        self.noise = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        torch.nn.init.normal_(self.noise.weight, mean=0, std=0.02) 
        self.ffn = FeedForward(
            d_model=self.hidden_size, inner_size=self.hidden_size * 4, dropout=self.dropout_prob
        )
        self.n_users = dataset.num(self.USER_ID)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.L = config["L"]
        self.beta = config["beta"]
        self.alpha = config["alpha"]
        self.rho = config["rho"]
        self.zeta = config["zeta"]
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.mlp_su = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.filter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.PMoE = PMoE(hidden_size=self.hidden_size)

        self.n_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attn_dropout_prob = 0.2
        self.layer_norm_eps = 1e-12
        self.linearattention = MultiHeadAttention(
            self.n_heads,
            self.hidden_size,
            self.hidden_dropout_prob,
            self.attn_dropout_prob,
            self.layer_norm_eps,
        )
        self.filter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)

        self.mlp_all = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 3),
            nn.Linear(self.hidden_size * 3, self.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 3, self.hidden_size * 3),
        )
        self.apply(self._init_weights)

        """granular preference interaction graph """
        # interaction
        interaction = dataset.inter_feat.interaction
        user_ids = interaction["user_id"]
        item_ids = interaction["item_id"]
        records = [{"user": u, "item": i} for u, i in zip(user_ids, item_ids)]

        # granular ball bridge
        self.path = f'dataset/{self.dataset_name}/granular_results_u_p{config["p"]}_i_p{config["p"]}.pth'
        (
            self.user_gblist,
            self.item_gblist,
            self.user_ball_embs,
            self.item_ball_embs,
        ) = self.load_and_process_granular_balls()
        self.num_user_balls = len(self.user_gblist)
        self.num_item_balls = len(self.item_gblist)
        with open(f"dataset/{self.dataset_name}/itm_emb_sasrec_seq.pkl", "rb") as f:
            i_embs = pickle.load(f)
            i_embs = torch.tensor(i_embs)
        with open(f"dataset/{self.dataset_name}/usr_emb_sasrec_seq.pkl", "rb") as f:
            u_embs = pickle.load(f)
            u_embs = torch.tensor(u_embs)
        zero_pad = torch.zeros(
            (1, u_embs.shape[1]), dtype=u_embs.dtype, device=u_embs.device
        )
        u_embs_padded = torch.cat([zero_pad, u_embs], dim=0)
        self.u_embs = u_embs_padded.cuda()
        zero_pad = torch.zeros(
            (1, i_embs.shape[1]), dtype=i_embs.dtype, device=i_embs.device
        )
        i_embs = torch.cat([zero_pad, i_embs], dim=0)
        self.i_embs = i_embs.cuda()
        self.user2ball = self.build_user2ball_tensor(
            self.user_gblist, self.n_users
        ).cuda()
        self.item2ball = self.build_user2ball_tensor(
            self.item_gblist, self.n_items
        ).cuda()

        # interagte
        ui_adj = self.create_unified_sparse_adjacency(
            records, self.user_gblist, self.item_gblist, True
        )
        norm_adj = self.normalize_graph_mat(ui_adj)
        self.sparse_norm_adj = self.convert_sparse_mat_to_tensor(norm_adj).cuda()
        """granular preference interaction graph """

        # use backbone as item_embedding
        self.item_embedding = nn.Embedding.from_pretrained(
            i_embs, freeze=False, padding_idx=0
        )

        self.align = nn.MSELoss()

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def create_unified_sparse_adjacency(
        self, records, user_gblist, item_gblist, self_connection=False
    ):
        """
        Construct a unified sparse adjacency matrix with the following nodes:
            - Users
            - Items
            - User-ball
            - Item-ball
        Returns:
            adj_total: Sparse adjacency matrix of size (n_users + n_items + num_user_balls + num_item_balls)^2
        """
        n_users = self.n_users
        n_items = self.n_items
        num_user_balls = len(user_gblist)
        num_item_balls = len(item_gblist)
        user_offset = 0
        item_offset = user_offset + n_users
        user_ball_offset = item_offset + n_items
        item_ball_offset = user_ball_offset + num_user_balls
        n_total = n_users + n_items + num_user_balls + num_item_balls
        rows = []
        cols = []

        # ---------- 1. user-item edge ----------
        for pair in records:
            u = pair["user"].item()
            i = pair["item"].item() + item_offset
            rows.extend([u, i])
            cols.extend([i, u])

        # ---------- 2. user-user ball edge ----------

        for ball_id, gb in enumerate(user_gblist):
            ball_node = user_ball_offset + ball_id
            if isinstance(gb, torch.Tensor):
                idxs = gb
            elif hasattr(gb, "indices"):
                idxs = gb.indices() if callable(gb.indices) else gb.indices
            else:
                raise TypeError(f"Unexpected type for gb: {type(gb)}")
            for u in idxs:
                u = u.item()
                rows.extend([u, ball_node])
                cols.extend([ball_node, u])

        # ---------- 3. item-item ball edge ----------
        for ball_id, gb in enumerate(item_gblist):
            ball_node = item_ball_offset + ball_id
            if isinstance(gb, torch.Tensor):
                idxs = gb
            elif hasattr(gb, "indices"):
                idxs = gb.indices() if callable(gb.indices) else gb.indices
            else:
                raise TypeError(f"Unexpected type for gb: {type(gb)}")
            for i in idxs:
                i = i.item() + item_offset
                rows.extend([i, ball_node])
                cols.extend([ball_node, i])
        data = np.ones(len(rows), dtype=np.float32)
        adj_total = sp.coo_matrix(
            (data, (rows, cols)), shape=(n_total, n_total), dtype=np.float32
        )

        if self_connection:
            adj_total += sp.eye(n_total)

        return adj_total.tocsr()

    def load_and_process_granular_balls(self):
        checkpoint = torch.load(self.path, map_location="cpu")
        user_gblist = checkpoint["user_gblist"]
        item_gblist = checkpoint["item_gblist"]
        user_ball_embs = checkpoint["user_ball_embs"].to(self.device)
        item_ball_embs = checkpoint["item_ball_embs"].to(self.device)
        print(f"Loaded granular results with params: {checkpoint['params']}")

        return user_gblist, item_gblist, user_ball_embs, item_ball_embs

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        rowsum[rowsum == 0] = 1e-9
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_sparse_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)

    def build_user2ball_tensor(self, user_gblist, n_users, device="cuda"):
        # Initialize the mapping, -1 indicates unassigned (optional)
        user2ball_tensor = torch.full((n_users,), -1, dtype=torch.long, device=device)

        for ball_idx, ball in enumerate(user_gblist):
            # Extract the list of user IDs
            if isinstance(ball, torch.Tensor):
                indices = ball.to(device)
            elif hasattr(ball, "indices"):
                indices = (
                    ball.indices()
                    if callable(ball.indices)
                    else torch.tensor(ball.indices, device=device)
                )
            else:
                raise TypeError(f"Unexpected type for ball: {type(ball)}")

            # Map these users to the current ball index
            user2ball_tensor[indices] = ball_idx
        return user2ball_tensor

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def apply_ib_guided_mask(self, emb, rho=0.3, method="softmax_max"):
        """
        Select the most confident granular ball embeddings (e.g., user_ball_embs) based on a masking model.

        Args:
            emb (Tensor): Granular ball embeddings with shape [N, D]
            rho (float): Disturbance ratio
            method (str): Method to determine confidence, one of 'softmax_max' / 'entropy' / 'norm'

        Returns:
            disturb_emb (Tensor): Disturbed embeddings
        """
        n, dim = emb.size()
        num_disturb = int(n * rho)
        device = emb.device

        with torch.no_grad():
            if method == "softmax_max":
                logits = self.proj(emb)
                probs = torch.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1).values  # [N]
            else:
                raise ValueError(f"Unknown method: {method}")

            # top-k most confident
            disturb_indices = torch.topk(
                confidence, k=num_disturb, largest=True
            ).indices
        disturb_emb = emb.clone()
        disturb_emb[disturb_indices] = (
            torch.randn((num_disturb, dim), device=device) * 0.01
        )
        return disturb_emb

    def graph_convolution(self, num_layer):

        # Initial input, which is freezed
        ui_temp = torch.cat(
            [self.u_embs, self.i_embs, self.user_ball_embs, self.item_ball_embs], dim=0
        )
        initial_emb = ui_temp
        all_embeddings = [initial_emb]

        for k in range(num_layer):
            # Apply perturbation at each layer
            ui_temp = self.apply_ib_guided_mask(ui_temp, rho=self.rho)

            # Graph convolution propagation
            ui_temp = torch.sparse.mm(self.sparse_norm_adj, ui_temp)

            all_embeddings.append(ui_temp)

        all_embeddings = torch.stack(
            all_embeddings, dim=1
        )  # shape: [n, num_layer+1, dim]
        output = torch.mean(all_embeddings, dim=1)  # shape: [n, dim]
        return output

    def forward(self, item_seq, item_seq_len):
        # Disturbance-Mutation Transfer Graph Network
        ui_embeds = self.graph_convolution(self.L)
        user_embedding = ui_embeds[0 : self.n_users]
        item_embedding = ui_embeds[self.n_users : self.n_users + self.n_items]

        item_balls = ui_embeds[self.n_users + self.n_items + self.num_user_balls :]
        user_balls = ui_embeds[
            self.n_users
            + self.n_items : self.n_users
            + self.n_items
            + self.num_user_balls
        ]

        # Disturbance-Enhanced Mixture of Experts
        Su, Su_p = self.DMoE(
            item_balls[self.item2ball[item_seq]],
            item_embedding[item_seq],
            self.item_embedding(item_seq),
            self.noise(item_seq),
        )

        # Hybrid Transition Tracker
        Fu = self.HTT(Su, Su_p)

        # Preference Fusion Predictor
        Sf = self.PFP(Su, Fu, Su_p)

        seq_output = self.gather_indexes(Sf, item_seq_len - 1)
        return (
            seq_output,
            user_balls,
            user_embedding,
        )

    def calculate_loss(self, interaction):
        user_ids = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, user_ball_embedding, user_embeds = self.forward(
            item_seq, item_seq_len
        )
        user_ball_idx = self.user2ball[user_ids]
        user_ball_emb = user_ball_embedding[user_ball_idx]

        # distillation losses
        loss_gd = self.align(seq_output, user_ball_emb.detach())
        loss_ud = self.align(seq_output, user_embeds[user_ids].detach())

        pos_items = interaction[self.POS_ITEM_ID]
        item_embeddings = self.item_embedding.weight

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = item_embeddings[pos_items]
            neg_items_emb = item_embeddings[neg_items]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = item_embeddings
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        pos_items_emb = item_embeddings[pos_items]
        return loss + self.beta * loss_ud + self.alpha * loss_gd

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _, _ = self.forward(item_seq, item_seq_len)

        item_embeddings = self.item_embedding.weight
        test_item_emb = item_embeddings[test_item]
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _, _ = self.forward(item_seq, item_seq_len)
        # item_embeddings = self.adapter(self.item_emb.weight)
        item_embeddings = self.item_embedding.weight
        test_items_emb = item_embeddings
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
    def DMoE(self, item_balls_seq_emb, item_granular_seq_emb, item_seq_emb, noise):
        # disturbance preference sequence
        item_emb = item_balls_seq_emb + item_seq_emb + self.zeta * noise
        item_emb_weight = self.filter(item_emb)
        item_emb_weight = torch.softmax(item_emb_weight, dim=-1)
        item_emb = item_emb_weight * item_emb
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        Su = self.linearattention(item_emb, item_emb)
        Su = self.LayerNorm(self.dropout(Su))
        # preference filtering MoE
        Sup = self.PMoE(item_emb, item_granular_seq_emb)
        return Su, Sup

    def HTT(self, Su, Sup):
        # transiton matrix
        Du = Sup - Su
        delta1 = Du[:, 1:, :] - Du[:, :-1, :]  # [B, n-1, d]  Δx_t - Δx_{t-1}
        Du = F.pad(delta1, (0, 0, 1, 0))  # [B, n, d]
        Du_high = self.mlp(Du)
        Su1 = self.mlp_su(Su)  # [B, n, d]
        diff_transition_matrix = torch.bmm(Du_high, Su1.transpose(1, 2))  # [B, n, n]
        score = torch.tanh(diff_transition_matrix)
        w = torch.softmax(score, dim=-1)  # [B, n, n]
        Su_t = torch.bmm(w, Su)  # [B, n, D]

        # magnitude matrix
        magnitude_matrix = torch.sigmoid(
            self.gate(torch.cat([Du_high, Su], dim=-1))
        )  # [B, n, D]
        magnitude = magnitude_matrix * Du_high
        Fu = self.LayerNorm(self.dropout(Su_t + magnitude))
        return Fu
    
    def PFP(self, Su, Fu, Su_p):
        weight = self.mlp_all(torch.concat([Su_p, Fu, Su], dim=-1))
        weight = weight.view(*weight.shape[:-1], 3, self.hidden_size)
        weight = torch.softmax(weight, dim=-2)
        weight_llm, weight_moe, weight_attn = (
            weight[:, :, 0, :],
            weight[:, :, 1, :],
            weight[:, :, 2, :],
        )
        Sf = weight_llm * Su_p + weight_moe * Fu + weight_attn * Su
        Sf = self.LayerNorm(self.dropout(Sf))
        Sf = self.ffn(Sf)
        return Sf


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)  # row-wise
        self.softmax_col = nn.Softmax(dim=-2)  # column-wise
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, q, input_tensor):
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)
        query_norm_inverse = 1 / torch.norm(elu_query, dim=3, p=2)  # (L2 norm)
        key_norm_inverse = 1 / torch.norm(elu_key, dim=2, p=2)
        normalized_query_layer = torch.einsum(
            "mnij,mni->mnij", elu_query, query_norm_inverse
        )
        normalized_key_layer = torch.einsum("mnij,mnj->mnij", elu_key, key_norm_inverse)
        context_layer = (
            torch.matmul(
                normalized_query_layer, torch.matmul(normalized_key_layer, value_layer)
            )
            / self.sqrt_attention_head_size
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + q)

        return hidden_states


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer"""

    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.lin1 = nn.Linear(input_size, output_size, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin2 = nn.Linear(input_size, output_size, bias=False)
        self.bias2 = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.LayerNorm = nn.LayerNorm((input_size))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class PMoE(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_experts=3,
        dropout_rate=0.1,
        noisy_gating=True,
        noise_epsilon=1e-2,
    ):
        super(PMoE, self).__init__()
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon
        self.n_heads = 4
        self.hidden_size = hidden_size
        self.layer_norm_eps = 1e-12
        self.experts = nn.ModuleList(
            [
                MultiHeadAttention(
                    self.n_heads,
                    self.hidden_size,
                    dropout_rate,
                    dropout_rate,
                    self.layer_norm_eps,
                )
                for i in range(num_experts)
            ]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_experts)]
        )

        # Gating网络
        self.gating_network = GatingNetwork(hidden_size, num_experts)
        self.w_noise = nn.Parameter(
            torch.zeros(hidden_size, num_experts)
        )  # 用于noisy gating

    def forward(self, q, kv):
        # 获取gating logits
        clean_logits = self.gating_network(q)

        # 加噪音到gating logits
        if self.noisy_gating:
            raw_noise_stddev = q @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = (
                clean_logits
                + torch.randn_like(clean_logits).to(q.device) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits
        expert_weights = F.softmax(logits, dim=-1).permute(0, 2, 1).unsqueeze(-1)

        expert_outputs = []
        for i in range(self.num_experts):
            output = self.dropouts[i](self.experts[i](q, kv))
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        output = torch.sum(expert_outputs * expert_weights, dim=1)
        return output
