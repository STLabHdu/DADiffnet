import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self, node_num, input_hidden, output_hidden, seq_len=12, horizon=12):
        super(BaseModel, self).__init__()
        self.node_num = node_num
        self.input_hidden = input_hidden
        self.output_hidden = output_hidden
        self.seq_len = seq_len
        self.horizon = horizon


    @abstractmethod
    def forward(self):
        raise NotImplementedError


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])