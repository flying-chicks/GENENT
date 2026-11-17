# encoding: utf-8

import torch
from torch.nn import ModuleList
from torch.nn import Sequential as Seq, ReLU, Linear as Lin, Softmax
from torch_geometric.nn import GINEConv, BatchNorm, global_mean_pool
from pathlib import Path


class MutagNet(torch.nn.Module):
    def __init__(self, input_dim=14, output_dim=2):
        super(MutagNet, self).__init__()

        conv_unit=2
        self.node_emb = Lin(input_dim, 32)
        self.edge_emb = Lin(3, 32)
        self.relu_nn = ModuleList([ReLU() for i in range(conv_unit)])

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(conv_unit):
            conv = GINEConv(nn=Seq(Lin(32, 75), self.relu_nn[i], Lin(75, 32)))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(32))
            self.relus.append(ReLU())

        self.lin1 = Lin(32, 16)
        self.relu = ReLU()
        self.lin2 = Lin(16, output_dim)
        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_pred(graph_x)

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        node_x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            node_x = conv(node_x, edge_index, edge_attr)
            node_x = ReLU(batch_norm(node_x))
        return node_x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.relu(self.lin1(graph_x))
        pred = self.lin2(pred)
        self.readout = self.softmax(pred)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

MUTAG_NET = MutagNet()
dir_path = Path(__file__).parent
# MUTAG_NET.load_state_dict(torch.load(dir_path / 'gnn_params/mutag_net_state_dict.pt'), strict=True)

from collections import OrderedDict

# 原始 state_dict（从文件加载的）
state_dict = torch.load(dir_path / 'gnn_params/mutag_net_state_dict.pt')

# 添加 'module.' 前缀
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # 在 batch_norms.X. 后面插入 module.
    # 例如: batch_norms.0.weight -> batch_norms.0.module.weight
    if k.startswith('batch_norms.'):
        parts = k.split('.')
        # parts = ['batch_norms', '0', 'weight']
        new_key = f"{parts[0]}.{parts[1]}.module.{'.'.join(parts[2:])}"
        new_state_dict[new_key] = v
    else:
        new_state_dict[k] = v

# 加载修改后的 state_dict
MUTAG_NET.load_state_dict(new_state_dict)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MUTAG_NET = MUTAG_NET.to(DEVICE)