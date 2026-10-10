'''Train a two-moons density model and inspect volume-preserving shear maps.'''

import argparse
import json
from pathlib import Path

import torch

from moonflow.pipeline import run_experiment
from moonflow.training import TrainConfig


def main() -> None:
    '''Read run settings and save the experiment outputs.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/two_moons'))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--patience', type=int, default=300)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--sample-seed', type=int, default=2026)
    parser.add_argument('--noise', type=float, default=0.08)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    torch.set_num_threads(args.threads)
    config = TrainConfig(n_layers=args.layers, hidden=args.hidden, learning_rate=args.learning_rate,
                         max_epochs=args.epochs, patience=args.patience, seed=args.model_seed)

    def progress(record: dict) -> None:
        '''Report post-update losses every 250 epochs.'''
        if record['epoch'] % 250 == 0:
            print(f"Epoch {record['epoch']}: train={record['train_nll']:.4f}, validation={record['val_nll']:.4f}", flush=True)

    summary = run_experiment(args.output_dir, data_dir=args.data_dir, config=config,
                             data_seed=args.data_seed, noise=args.noise, sample_seed=args.sample_seed,
                             progress=progress)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
