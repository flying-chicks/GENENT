# encoding: utf-8
import torch
from torch.nn import Linear, ReLU, ModuleList, Softmax
from torch_geometric.nn import global_mean_pool ,LEConv
import torch.nn.functional as F
from pathlib import Path


class BA2MotifNet(torch.nn.Module):
    def __init__(self, num_unit):
        super().__init__()

        self.num_unit = num_unit

        self.node_emb = Linear(5, 64)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(num_unit):
            conv = LEConv(in_channels=64, out_channels=64)
            self.convs.append(conv)
            self.relus.append(ReLU())

        self.lin1 = Linear(64, 128)
        self.relu = ReLU()
        self.lin2 = Linear(128, 3)
        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_pred(graph_x)

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        node_x = self.node_emb(x)
        node_x = F.dropout(node_x, p=0.4)
        for conv, ReLU in \
                zip(self.convs, self.relus):
            node_x = conv(x=node_x, edge_index=edge_index, edge_weight=edge_attr)
            node_x = ReLU(node_x)
        return node_x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.relu(self.lin1(graph_x))
        pred = self.lin2(pred)
        self.readout = pred
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


dir_path = Path(__file__).parent
state_dict = torch.load(dir_path / 'gnn_params/ba2_net_state_dict.pt')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_LAYERS = 3
BA2MOTIF_NET = BA2MotifNet(NUM_LAYERS).to(DEVICE)
BA2MOTIF_NET.load_state_dict(state_dict)
