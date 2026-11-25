"""Train the MUTAG discrete diffusion model for paired graph sampling."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'saved_models/mutag_diffusion.ckpt'


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['MUTAG'], default='MUTAG')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(['--' + a if '=' in a and not a.startswith('-') else a for a in argv])
    if min(args.epochs, args.batch_size) < 1:
        parser.error('epochs and batch_size must be positive')
    return args


def diffusion_graph(graph):
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.utils import remove_self_loops, to_undirected

    index, bonds = remove_self_loops(graph.edge_index, graph.edge_attr)
    index, bonds = to_undirected(index, bonds, num_nodes=graph.num_nodes, reduce='max')
    return Data(x=graph.x.float(), edge_index=index, edge_attr=F.pad(bonds.float(), (1, 0)),
                y=graph.x.new_empty((1, 0)))


def train(args):
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader
    from src.datasets.mutag_dataset import MutagenicityDataset
    from src.diffusion.distributions import DistributionNodes
    from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
    from src.diffusion_model_discrete import DiscreteDenoisingDiffusion

    pl.seed_everything(args.seed)
    dataset = MutagenicityDataset(str(ROOT / 'data/MUTAG'), mode='training', transform=diffusion_graph)
    sizes = Counter()
    nodes, edges = torch.zeros(14), torch.zeros(4)
    for graph in dataset:
        n = graph.num_nodes
        sizes[n] += 1
        nodes += graph.x.sum(0)
        edges += graph.edge_attr.sum(0)
        edges[0] += n * (n - 1) - graph.num_edges

    cfg = OmegaConf.create({'model': OmegaConf.load(ROOT / 'configs/model/discrete.yaml'),
                            'train': OmegaConf.load(ROOT / 'configs/train/train_default.yaml'),
                            'general': {'name': 'mutag_diffusion', 'log_every_steps': 50, 'number_chain_steps': 0}})
    info = SimpleNamespace(output_dims={'X': 14, 'E': 4, 'y': 0}, max_n_nodes=max(sizes),
                           nodes_dist=DistributionNodes(dict(sizes)), node_types=nodes, edge_types=edges)
    features = ExtraFeatures(cfg.model.extra_features, info) if cfg.model.extra_features else DummyExtraFeatures()
    from src import utils
    example = next(iter(DataLoader(dataset, batch_size=1)))
    dense, mask = utils.to_dense(example.x, example.edge_index, example.edge_attr, example.batch)
    extra = features(dict(X_t=dense.X, E_t=dense.E, y_t=example.y, node_mask=mask))
    info.input_dims = {'X': 14 + extra.X.size(-1), 'E': 4 + extra.E.size(-1), 'y': 1 + extra.y.size(-1)}
    model = DiscreteDenoisingDiffusion(cfg, info, None, None, None, features, DummyExtraFeatures())
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=1, logger=False, enable_checkpointing=False, limit_val_batches=0,
                         num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=args.batch_size, shuffle=True))
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(CHECKPOINT), weights_only=True)
    print(f'Saved trained diffusion checkpoint: {CHECKPOINT}')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    options = parse_args(sys.argv[1:])
    try:
        train(options)
    except ModuleNotFoundError as exc:
        sys.exit(f'Missing training dependency: {exc.name}. Activate the diffusion model environment.')
