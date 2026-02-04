import torch
from torch import nn

from .transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, \
    TransformerDecoderLayer


class FeatureTransformerBlock(nn.Module):
    def __init__(self, his_len, input_len, nhead=1, dropout=0.1):
        super().__init__()

        self.his_len = his_len
        self.input_len = input_len
        self.nhead = nhead
        self.drop_out = dropout

        self.long_linear = nn.Sequential(
            nn.Linear(in_features=self.his_len, out_features=self.input_len, bias=True),
        )

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=self.input_len, nhead=1, dim_feedforward=4 * self.input_len,
                                    batch_first=True), num_layers=self.nhead)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model=self.input_len, nhead=1, dim_feedforward=4 * self.input_len,
                                    batch_first=True), num_layers=self.nhead)

    def forward(self, long_input_data,input_data):
        long_input_data_emb = []
        long_input_data = self.long_linear(long_input_data)
        long_input_data_emb.append(long_input_data)
        input_data_en = []
        input_data_de = []
        input_data_en.append(self.encoder(input_data))
        input_data_de.append(self.decoder(input_data, input_data_en[0]))

        time_series_emb = [torch.cat(long_input_data_emb + input_data_en + input_data_de, dim=2)]
        return time_series_emb


class GraphExtractBlock(nn.Module):
    def __init__(self, st_dim, input, graph_num,
                 output_dim, dropout=0.1, **model_args):
        super().__init__()

        self.st_dim = st_dim
        self.input_len = input
        self.graph_out_dim = model_args["graph_out_dim"]
        self.graph_num = graph_num
        self.graph_num_step = model_args["graph_num_step"]
        self.output_dim = output_dim
        self.graph_num_layer = model_args["graph_num_layer"]
        self.output_len = model_args["output_len"]
        self.graph_layers = nn.ModuleList([
            GraphExt(
                input_dim=self.st_dim + self.input_len + self.input_len + self.input_len,
                hidden_dim=self.st_dim + self.input_len + self.input_len + self.input_len,
                out_dim=self.graph_out_dim,
                graph_num=self.graph_num,
                next=False, **model_args)
        ])
        for _ in range(self.graph_num_step - 1):
            self.graph_layers.append(
                GraphExt(input_dim=self.st_dim + self.graph_out_dim,
                          hidden_dim=self.st_dim + self.graph_out_dim,
                          out_dim=self.graph_out_dim,
                          graph_num=self.graph_num,
                          next=True, **model_args)
            )
        if self.graph_num_step > 1:
            self.regression_layer = nn.Sequential(
                *[MultiLayerPerceptron(input_dim=self.output_dim,
                                       hidden_dim=self.output_dim,
                                       dropout=dropout)
                  for _ in range(self.graph_num_layer)],
                nn.Linear(in_features=self.output_dim, out_features=self.output_len, bias=True),
            )

    def forward(self, history_data,
                time_series_emb,
                ori_emb, augment_emb):

        predicts = []
        predict_emb = []
        hidden_ori_emb = []
        hidden_augment_emb = []
        for index, layer in enumerate(self.graph_layers):
            predict, hidden_ori, hidden_augment, \
            ori_emb_forward, augment_emb_forward = layer(history_data, time_series_emb, predict_emb,
                                                                ori_emb, augment_emb,
                                                                hidden_ori_emb, hidden_augment_emb)
            predicts.append(predict)
            predict_emb = [predict]
            time_series_emb = []
            hidden_ori_emb = hidden_ori
            hidden_augment_emb = hidden_augment

            ori_emb = ori_emb_forward
            augment_emb = augment_emb_forward

        predicts = torch.cat(predicts, dim=2)
        if self.graph_num_step > 1:
            predicts = self.regression_layer(predicts)

        return predicts


class GraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, graph_num_layer, dropout):
        super().__init__()

        self.graph_layer = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=input_dim,
                                   hidden_dim=hidden_dim,
                                   dropout=dropout)
              for _ in range(graph_num_layer)],
        )

    def forward(self, time_series_emb, predict_emb,
                ori_emb, tem_emb):
        node_emb = torch.cat(time_series_emb + predict_emb + ori_emb + tem_emb, dim=2)
        hidden_ori = self.graph_layer(node_emb)
        return hidden_ori

class GraphAttModel(nn.Module):
    def __init__(self, graph_dim, nhead, node_dim):
        super().__init__()

        self.att_layer = nn.MultiheadAttention(embed_dim=graph_dim,
                                                 num_heads=nhead,
                                                 batch_first=True)
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=graph_dim, out_features=node_dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, emb, hidden_emb):
        emb = emb[0]
        hidden_ori_emb = hidden_emb[0]
        hidden_ori_emb = \
            self.att_layer(hidden_ori_emb, hidden_ori_emb, hidden_ori_emb)[0]
        hidden_ori_emb = self.fc_layer(hidden_ori_emb)
        emb = [emb * hidden_ori_emb]
        return emb


class Time_embedding(nn.Module):
    def __init__(self, if_time_in_day, if_day_in_week, time_of_day_size,
                 day_of_week_size, temp_dim_tid, temp_dim_diw):
        super().__init__()
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

    def forward(self, history_data):
        tem_emb = []
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1] * self.time_of_day_size
            tem_emb.append(self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)])
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2] * self.day_of_week_size
            tem_emb.append(self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)])
        return tem_emb

class MultiMLP(nn.Module):
    def __init__(self, graph_num, graph_dim, dropout, graph_num_layer, out_dim):
        super().__init__()
        self.graph_num = graph_num
        self.graph_dim = graph_dim
        self.dropout = dropout
        self.graph_num_layer = graph_num_layer
        self.out_dim = out_dim
        self.mlp_layer = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=self.graph_num * self.graph_dim,
                                   hidden_dim=self.graph_num * self.graph_dim,
                                   dropout=self.dropout)
              for _ in range(self.graph_num_layer)],
            nn.Linear(in_features=self.graph_num * self.graph_dim, out_features=self.out_dim, bias=True),
        )
    def forward(self, history_data):
        return self.mlp_layer(history_data)


class GraphExt(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, graph_num, next, **model_args):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.graph_num = graph_num
        self.next = next

        self.if_augment = model_args["if_augment"]
        self.node_dim = model_args["node_dim"]
        self.nhead = model_args["nhead"]

        self.graph_num_layer = model_args["graph_num_layer"]
        self.graph_dim = model_args["graph_dim"]
        self.dropout = model_args["graph_dropout"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.Time_embedding = Time_embedding(self.if_time_in_day, self.if_day_in_week, self.time_of_day_size,
                                             self.day_of_week_size, self.temp_dim_tid, self.temp_dim_diw)

        if self.graph_num > 1:
            self.graph_layer = GraphModel(self.input_dim, self.hidden_dim, self.graph_num_layer, self.dropout)
            self.graph_ori_linear = nn.Linear(in_features=hidden_dim, out_features=self.graph_dim,
                                                   bias=True)
            if self.if_augment:
                self.graph_augment_linear = nn.Linear(in_features=self.hidden_dim, out_features=self.graph_dim,
                                                        bias=True)
            self.graph_model = MultiMLP(self.graph_num, self.graph_dim, self.dropout,
                                         self.graph_num_layer, self.out_dim)

        else:
            self.graph_model = nn.Sequential(
                *[MultiLayerPerceptron(input_dim=self.input_dim,
                                       hidden_dim=self.hidden_dim,
                                       dropout=self.dropout)
                  for _ in range(self.graph_num_layer)],
                nn.Linear(in_features=self.hidden_dim, out_features=self.node_dim, bias=True)
            )
            self.graph_linear = nn.Linear(in_features=self.node_dim, out_features=self.out_dim, bias=True)

        if self.next:
            self.graph_att_layer = GraphAttModel(self.graph_dim, self.nhead, self.node_dim)
            if self.if_augment:
                self.augment_graph_att_layer = GraphAttModel(self.graph_dim, self.nhead, self.node_dim)

    def forward(self, history_data,
                time_series_emb, predict_emb,
                ori_emb, augment_emb,
                hidden_ori_emb, hidden_augment_emb):

        if self.next:
            ori_emb = self.graph_att_layer(ori_emb, hidden_ori_emb)
            if self.if_augment:
                augment_emb = self.augment_graph_att_layer(augment_emb, hidden_augment_emb)

        tem_emb = self.Time_embedding(history_data)

        if self.graph_num > 1:
            hidden_augment = []
            hidden_ori = self.graph_layer(time_series_emb, predict_emb, ori_emb, tem_emb)
            hidden_ori = [self.graph_ori_linear(hidden_ori)]
            if self.if_augment:
                hidden_augment = self.graph_layer(time_series_emb, predict_emb, augment_emb, tem_emb)
                hidden_augment = [self.graph_augment_linear(hidden_augment)]
            hidden = torch.cat(hidden_ori + hidden_augment, dim=2)
            predict = self.graph_model(hidden)
            return predict, hidden_ori, hidden_augment, ori_emb, augment_emb
        else:
            hidden = torch.cat(
                time_series_emb + predict_emb + ori_emb + augment_emb + tem_emb, dim=2)
            hidden = self.graph_model(hidden)
            predict = self.graph_linear(hidden)
        return predict, [hidden], [hidden], ori_emb, augment_emb


class GraphMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x + self.fc2(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(input_data)  # MLP
        hidden = hidden + input_data  # residual
        return hidden

