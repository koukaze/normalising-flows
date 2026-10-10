'''Generate two-moons examples and read external coordinate splits.'''

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MoonSplit:
    '''Store two-dimensional points and optional labels for visualization.

    Args
        points (ndarray): Finite coordinates, shape (n_points, 2).
        labels (ndarray or None): Optional binary class labels, shape (n_points,).
    '''

    points: np.ndarray
    labels: np.ndarray | None = None

    def tensor(self) -> torch.Tensor:
        '''Return the coordinates as a CPU float32 tensor.'''
        return torch.tensor(self.points, dtype=torch.float32)


def make_moons_splits(
    n_train: int = 800,
    n_val: int = 100,
    n_test: int = 100,
    noise: float = 0.08,
    seed: int = 42,
) -> dict[str, MoonSplit]:
    '''Generate independently perturbed, balanced two-moons data splits.

    Args
        n_train (int): Number of training points, at least four.
        n_val (int): Number of validation points, at least four.
        n_test (int): Number of test points, at least four.
        noise (float): Standard deviation of independent Gaussian coordinate noise.
        seed (int): SeedSequence entropy for three independent random streams.

    Returns
        dict: Training, validation and test MoonSplit objects.
    '''
    counts = (n_train, n_val, n_test)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 4 for n in counts):
        raise ValueError('Each split must contain at least four points.')
    if not np.isfinite(noise) or noise < 0:
        raise ValueError('noise must be finite and non-negative.')
    streams = np.random.SeedSequence(seed).spawn(3)
    result = {}
    for name, count, stream in zip(('train', 'val', 'test'), counts, streams):
        rng = np.random.default_rng(stream)
        first, second = count // 2, count - count // 2
        upper_angle = np.linspace(0, np.pi, first)
        lower_angle = np.linspace(0, np.pi, second)
        points = np.concatenate((
            np.column_stack((np.cos(upper_angle), np.sin(upper_angle))),
            np.column_stack((1 - np.cos(lower_angle), 0.5 - np.sin(lower_angle))),
        ))
        labels = np.r_[np.zeros(first, dtype=int), np.ones(second, dtype=int)]
        points += rng.normal(0, noise, size=points.shape)
        order = rng.permutation(count)
        result[name] = MoonSplit(points[order], labels[order])
    return result


def load_moons_splits(data_dir: str | Path) -> dict[str, MoonSplit]:
    '''Load finite coordinate arrays from external train, validation and test CSVs.

    Args
        data_dir (str or Path): Directory containing moons_train.csv, moons_val.csv and moons_test.csv.

    Returns
        dict: Named MoonSplit objects; class labels are optional and are never model inputs.
    '''
    result = {}
    for name in ('train', 'val', 'test'):
        path = Path(data_dir) / f'moons_{name}.csv'
        table = np.genfromtxt(path, delimiter=',', names=True, dtype=float)
        names = table.dtype.names or ()
        if not {'x1', 'x2'}.issubset(names):
            raise ValueError(f'{path.name} must contain x1 and x2 columns.')
        points = np.column_stack((np.atleast_1d(table['x1']), np.atleast_1d(table['x2'])))
        if len(points) == 0 or not np.isfinite(points).all():
            raise ValueError(f'{path.name} contains missing or non-finite coordinates.')
        labels = None
        if 'class' in names:
            labels = np.atleast_1d(table['class'])
            if not np.isin(labels, [0, 1]).all():
                raise ValueError(f'{path.name} class labels must be zero or one.')
            labels = labels.astype(int)
        result[name] = MoonSplit(points, labels)
    return result


def save_moons_splits(splits: dict[str, MoonSplit], data_dir: str | Path) -> None:
    '''Save coordinate splits with optional class labels as ordinary CSV files.

    Args
        splits (dict): Training, validation and test MoonSplit objects.
        data_dir (str or Path): Destination for moons_train.csv, moons_val.csv and moons_test.csv.
    '''
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        values = split.points
        header, fmt = 'x1,x2', ['%.17g', '%.17g']
        if split.labels is not None:
            values = np.column_stack((values, split.labels))
            header, fmt = 'x1,x2,class', ['%.17g', '%.17g', '%d']
        np.savetxt(data_dir / f'moons_{name}.csv', values, delimiter=',', header=header, comments='', fmt=fmt)
