"""Generate paired control and treatment graphs from a trusted MUTAG diffusion checkpoint."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Example: python src/generate.py n_pairs=1000')
    parser.add_argument('--n_pairs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'saved_models/mutag_diffusion.ckpt')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/graph_pairs.pt')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(['--' + a if '=' in a and not a.startswith('-') else a for a in argv])
    if min(args.n_pairs, args.batch_size) < 1:
        parser.error('n_pairs and batch_size must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Missing diffusion checkpoint: {args.checkpoint}. Supply checkpoint=/path/to/model.ckpt')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}. Choose a new output=path.')
    return args


def select_observation(dataset):
    import torch

    for index, graph in enumerate(dataset):
        x = graph.x.argmax(dim=-1)
        e = torch.zeros((len(x), len(x)), dtype=torch.long)
        i, j = graph.edge_index
        # Real bonds use 1..3; zero denotes no edge in the diffusion sampler.
        e[i, j] = e[j, i] = graph.edge_attr.argmax(dim=-1) + 1
        for nitrogen in (x == 4).nonzero(as_tuple=True)[0].tolist():
            oxygen = ((e[nitrogen] != 0) & (x == 1)).nonzero(as_tuple=True)[0]
            if len(oxygen) >= 2:
                region = torch.zeros_like(x, dtype=torch.bool)
                region[nitrogen] = True
                region[oxygen[:2]] = True
                return index, x, e, region
    raise ValueError('No N with two distinct O neighbors in the MUTAG test split.')


def generate(args):
    import torch
    from src.datasets.mutag_dataset import MutagenicityDataset
    from src.diffusion_model_discrete import DiscreteDenoisingDiffusion

    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        str(args.checkpoint), map_location='cpu', train_metrics=None,
        sampling_metrics=None, visualization_tools=None,
    )
    if (model.Xdim_output, model.Edim_output) != (14, 4):
        raise ValueError('Expected a MUTAG checkpoint with 14 atom types and 4 edge types (including no edge).')
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    dataset = MutagenicityDataset(str(ROOT / 'data/MUTAG'), mode='testing')
    index, x, e, region = select_observation(dataset)
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    pairs = []
    for start in range(0, args.n_pairs, args.batch_size):
        count = min(args.batch_size, args.n_pairs - start)
        pairs.extend(model.sample_graph_pairs(x, e, region, num_pairs=count, generator=generator))
        print(f'Generated {len(pairs)}/{args.n_pairs} pairs', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(pairs=pairs, node_types=x, edge_types=e, exp_nodes=region,
                    test_index=index, seed=args.seed), args.output)
    print(f'Saved {len(pairs)} pairs to {args.output}')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    options = parse_args(sys.argv[1:])
    try:
        generate(options)
    except ModuleNotFoundError as exc:
        sys.exit(f'Missing project dependency: {exc.name}. Run in the diffusion model training environment.')
