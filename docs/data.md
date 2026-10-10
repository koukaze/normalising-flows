# Data

## Synthetic two-moons experiment

The default experiment generates a two-dimensional dataset directly with NumPy. Each split contains two half-circles with equal class sizes, or a difference of one point when the split size is odd. For angles equally spaced on $[0,\pi]$, including the endpoints, their coordinates are

$$
(x_1,x_2)=(\cos\theta,\sin\theta)
$$

and

$$
(x_1,x_2)=(1-\cos\theta,0.5-\sin\theta).
$$

Independent Gaussian noise is added to both coordinates, then the rows are shuffled. The default noise standard deviation is **0.08**, with **800 training**, **100 validation** and **100 test** points and random seed **42**. Each split uses a separate random stream derived from the seed through NumPy `SeedSequence`. Coordinates and noise are dimensionless.

The generation code is included in the repository. No data download is required. This is a synthetic density-estimation example; its results are defined by the generator settings, split sizes and training configuration.

The model learns the joint density of `x1` and `x2`. Moon labels are used only for colours in diagnostic plots, never as training inputs or targets. Validation likelihood selects the checkpoint; the test split is used for final evaluation.

## External CSV inputs

A data directory may instead contain:

| File | Contents | Coursework split size |
|---|---|---:|
| `moons_train.csv` | Training points | 800 |
| `moons_val.csv` | Validation points | 100 |
| `moons_test.csv` | Test points | 100 |

The coursework CSVs use the header `x1,x2,class`; the loader also accepts `x1,x2` without labels. `x1` and `x2` must be finite numeric coordinates. When present, `class` must be 0 or 1 and identifies the two moons for plotting. Keep the split files separate. See [Running the project](../RUNNING.md) for the external-data command.

The Cambridge M2 coursework supplied these three CSV inputs. They are not distributed with this repository. The synthetic generator defines its own example dataset and does not reproduce the supplied CSV rows.

Keep external data outside the repository. Generated outputs record which dataset was used, so metrics from different inputs can be distinguished.
