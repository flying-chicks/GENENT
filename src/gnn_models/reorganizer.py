import torch
import copy
from tqdm import tqdm
from torch_geometric.data import DataLoader


def filter_correct_data(trained_gnn, loader, flag='Training'):
    """
    筛选出被 GNN 模型正确分类的图数据。

    参数:
        trained_gnn: 训练好的图神经网络模型。
        loader: 数据加载器。
        flag: 阶段标识，默认为 'Training'。

    返回:
        loader: 仅包含正确分类样本的新数据加载器。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_mask = torch.zeros(len(loader), dtype=torch.bool)
    idx = 0
    for g in tqdm(loader, total=len(loader)):
        g.to(device)
        trained_gnn(g.x, g.edge_index, g.edge_attr, g.batch)
        if g.y == trained_gnn.readout.argmax(dim=1):
            graph_mask[idx] = True
        idx += 1

    loader = DataLoader(loader.dataset[graph_mask], batch_size=1, shuffle=False)
    print("number of graphs in the %s: %4d" % (flag, graph_mask.nonzero().size(0)))
    return loader


def filter_correct_data_batch(trained_gnn, loader, flag='training', batch_size=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_mask = []
    for g in tqdm(loader, total=len(loader)):
        g.to(device)
        trained_gnn(g.x, g.edge_index, g.edge_attr, g.batch)

        tmp = (g.y == trained_gnn.readout.argmax(dim=1))
        graph_mask += tmp.tolist()

    # must convert the graph_mask to the boolean tensor.
    graph_mask = torch.BoolTensor(graph_mask)

    shuffle_flag = False
    if flag == 'training':
        shuffle_flag = True

    loader = DataLoader(loader.dataset[graph_mask], batch_size=batch_size, shuffle=shuffle_flag)
    print("number of graphs in the %s:%4d" % (flag, sum(graph_mask)))
    return loader
