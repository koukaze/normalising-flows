# Running the project

Run commands from the repository root. Training uses CPU tensors; a GPU is not required.

## Installation

Use Python 3.11 or newer. The recorded example was run with Python 3.13.7.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`requirements.txt` pins the versions used for the example. For a development environment with tests and notebooks:

```bash
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name moonflow --display-name "Python (moonflow)"
```

Select **Python (moonflow)** when opening the notebooks.

## Complete experiment

```bash
python scripts/train_flow.py --output-dir outputs/two_moons
```

This generates synthetic train, validation and test splits, trains an affine coupling flow, restores the weights with the lowest validation NLL, and evaluates those weights on the test split. It then saves samples, numerical diagnostics and figures. The command uses one CPU thread for a reproducible example; `--threads` changes this setting.

For a short check of the complete workflow:

```bash
python scripts/train_flow.py --epochs 20 --patience 10 --output-dir outputs/smoke
```

This short run checks execution and is not intended to learn the two-moons density accurately. Every run has its own cosine learning-rate schedule, so changing `--epochs` changes the optimisation trajectory.

Use a separate output directory for each configuration. Reusing a directory replaces files with the same names. The output directory contains:

| Output | Contents |
|---|---|
| `summary.json` | Selected-model NLLs, configuration, seeds, data hashes, numerical errors and software versions |
| `training_curves.json` | Post-update training/validation NLL and the learning rate used at each epoch |
| `flow.pt` | Selected weights, architecture, training settings and metadata |
| `data/moons_*.csv` | Generated synthetic inputs, when the generator is used |
| `samples.npz` | Model samples and paired samples at shear coefficients −1 and +1 |
| `jacobian_checks.npz` | Analytic, autograd and finite-difference inverse log-determinants |
| `density.png` / `.svg` | Training coordinates, fitted density and model samples |
| `training.png` / `.svg` | Learning curves and selected epoch |
| `shear_family.png` / `.svg` | The same model samples transformed by different shears |

All output directories are ignored by Git. The figures and metrics in `assets/` document the README example.

## External data

To use existing split files with `x1,x2` and optional `class` columns:

```bash
python scripts/train_flow.py \
  --data-dir /path/to/moons-data \
  --output-dir outputs/external_data
```

The directory must contain `moons_train.csv`, `moons_val.csv` and `moons_test.csv`. Coordinates are used as supplied, without rescaling or reshuffling. Class labels affect plots only. External files are read in place and are not copied into the output directory. `--data-seed` and `--noise` apply only to generated inputs. See [Data](docs/data.md) for the geometry and input requirements.

## Notebooks

```bash
python -m jupyterlab
```

- [01_flow_checks.ipynb](notebooks/01_flow_checks.ipynb) checks inverse maps, Jacobian determinants, densities and the idealised operation count.
- [02_density_and_shear.ipynb](notebooks/02_density_and_shear.ipynb) runs a short synthetic training example, inspects density and learning curves, and applies a family of invertible shears. A small-subset fit demonstrates training capacity without estimating generalisation.

The second notebook trains for up to 600 epochs by default. Its displayed metrics therefore belong to that notebook run, not the longer CLI example. To inspect a completed CLI run instead, set its output directory before launching Jupyter:

```bash
FLOW_RESULTS_DIR=outputs/two_moons python -m jupyterlab
```

## Using a trained flow

```python
import torch
from moonflow.training import load_checkpoint
from moonflow.transforms import family_log_prob, g_alpha

flow, config, metadata = load_checkpoint("outputs/two_moons/flow.pt")
generator = torch.Generator().manual_seed(2026)
with torch.no_grad():
    points = flow.sample(1000, generator=generator)
    log_density = flow.log_prob(points)
    sheared, _ = g_alpha(points, alpha=1.0)
    sheared_log_density = family_log_prob(flow, sheared, alpha=1.0)
```

The last two density arrays agree up to floating-point precision because the shear has unit determinant. The flow itself generally changes volume.

## Validation and reproducibility

```bash
python -m pytest
python -m compileall -q src scripts
```

[Numerical validation](docs/validation.md) explains the mathematical references and workflow checks. Training and generated data have separate seeds; sample generation uses its own random generator. The recorded run includes input hashes and library versions. Exact floating-point results can differ across software versions and hardware.

For additional CLI settings:

```bash
python scripts/train_flow.py --help
```
