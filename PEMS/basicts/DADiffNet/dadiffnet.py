import numpy as np
import torch
from torch import nn

from .bfs_block import get_link_list, get_tree_emb_link_list
from .mlp import MultiLayerPerceptron, GraphMLP, FeatureTransformerBlock, GraphExtractBlock
device = 'cuda'


class DADiffNet(nn.Module):

    def __init__(self, adj_mx, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.output_len = model_args["output_len"]
        self.his_len = 288

        self.tree_emb_len_list = model_args["tree_emb_len_list"]

        self.if_use_augment = model_args["if_augment"]
        self.if_use_diff = model_args["if_diff"]

        self.graph_num_step = model_args["graph_num_step"]
        self.graph_num_layer = model_args["graph_num_layer"]
        self.graph_dim = model_args["graph_dim"]
        self.graph_out_dim = model_args["graph_out_dim"]
        self.dropout = model_args["graph_dropout"]

        self.adj_mx = adj_mx
        self.node_dim = model_args["node_dim"]
        self.nhead = model_args["nhead"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]

        self.augment_adj_len = []
        self.augment_adj_mx = []

        if self.if_use_diff:
            self.input_len = self.input_len - 1
            self.his_len = 287

        if self.if_use_augment:
            adj_link_list = get_link_list(self.adj_mx[0], self.num_nodes)
            for l in self.tree_emb_len_list:
                emb_len, temp_emb = get_tree_emb_link_list(adj_link_list, l, self.num_nodes, "max", device)
                self.augment_adj_len.append(emb_len)
                self.augment_adj_mx.append(temp_emb)
            self.adj_mx_augment_encoder = nn.Sequential(
                GraphMLP(input_dim=self.augment_adj_len[0], hidden_dim=self.node_dim)
            )

        self.graph_num = 1 + 1 * int(self.if_use_augment)

        self.st_dim = (self.graph_num > 0) * self.node_dim + \
                      self.if_time_in_day * self.temp_dim_tid + \
                      self.if_day_in_week * self.temp_dim_diw

        self.output_dim = self.graph_num_step * self.graph_out_dim
        self.adj_mx_ori_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
        )

        self.FeatureTransformerBlock = FeatureTransformerBlock(self.his_len, self.input_len)
        self.GraphExtractBlock = GraphExtractBlock(self.st_dim, self.input_len, self.graph_num,
                                                   self.output_dim, **model_args)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor = None,
                batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:

        his_input_data = long_history_data[..., 0].transpose(1, 2)
        input_data = history_data[..., 0].transpose(1, 2)
        batch_size, num_nodes, _ = input_data.shape

        if self.if_use_diff:
            his_input_data = his_input_data[..., 1:] - his_input_data[..., :-1]
            input_data = input_data[..., 1:] - input_data[..., :-1]

        time_series_emb = self.FeatureTransformerBlock(his_input_data, input_data)

        ori_emb = []
        ori = self.adj_mx[0].to(device)
        ori = self.adj_mx_ori_encoder(ori.unsqueeze(0)).expand(batch_size, -1, -1)
        ori_emb.append(ori)

        augment_emb = []
        if self.if_use_augment:
            augment = self.augment_adj_mx[0].to(device)
            augment = self.adj_mx_augment_encoder(augment.unsqueeze(0)).expand(batch_size, -1, -1)
            augment_emb.append(augment)

        predicts = self.GraphExtractBlock(history_data, time_series_emb, ori_emb, augment_emb)

        if self.if_use_diff:
            trend = history_data[:, -1:,:, 0].transpose(1, 2).expand(-1, -1, self.output_len)
            predicts = predicts + trend

        return predicts.transpose(1, 2).unsqueeze(-1)



