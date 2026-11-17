# import torch

from torch_geometric.data import DataLoader
from reorganizer import  filter_correct_data, filter_correct_data_batch
from src.datasets.mutag_dataset import MutagenicityDataset
from src.gnn_models.mutag_gnn import MUTAG_NET

train_dataset = MutagenicityDataset('Data/MUTAG', mode='training')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# refine the datasets and data loaders
train_dataset, train_loader = filter_correct_data_batch(MUTAG_NET, train_dataset, train_loader, 'training', batch_size=4)