# Numerical validation

Run the automated checks from the repository root:

```bash
python -m pytest
```

The tests cover mathematical identities and the training/data workflow. The experiment also writes numerical diagnostics for its selected model, so implementation tests and fitted-model checks can be examined separately.

## Change of variables

The tests check both $f^{-1}(f(z))=z$ and $f(f^{-1}(x))=x$, along with cancellation of forward and inverse log-determinants. Fixed coordinates of each coupling layer must remain unchanged. Analytic log-determinants are compared with independently formed Jacobians from automatic differentiation and central finite differences.

Two reference distributions provide additional checks:

- With every network parameter set to zero, the full flow is the identity and its density must equal the standard multivariate Gaussian.
- Applying a shear to that identity flow must produce a Gaussian with covariance $\begin{pmatrix}1+\alpha^2&\alpha\\\alpha&1\end{pmatrix}$. The induced density is compared with PyTorch's multivariate Gaussian implementation.

The shear tests also verify its explicit Jacobian, unit determinant, inverse and composition with a nontrivial flow. Sampling tests check reproducibility with a local generator, dtype/device consistency and supported batch shapes. A gradient check verifies that the likelihood supplies finite gradients to all model parameters.

## Training and saved models

The training tests check that a fixed seed reproduces a fit without altering the caller's input tensors or random-number state. Both loss curves are evaluated after an optimiser update. A controlled early-stopping test verifies restoration of the lowest-validation-NLL weights, and a small-subset fit checks that optimisation can reduce training NLL.

Checkpoint tests compare densities before and after saving/loading. The loader checks the architecture, coupling masks, tensor shapes and dtypes, finite values, epoch metadata and configuration. It uses PyTorch's restricted `weights_only=True` loading mode. Invalid input features and inconsistent checkpoint contents are rejected.

## Data

Data tests check deterministic generation, distinct random streams for the splits, CSV save/load consistency, optional plotting labels and rejection of non-finite coordinates. The external-data loader does not use labels as model inputs. Train, validation and test features remain separate throughout fitting and evaluation.

## Fitted-model diagnostics

Each full experiment records:

- float32 round-trip and log-determinant cancellation errors over all training points;
- analytic, automatic-differentiation and central-difference inverse log-determinants for up to the first 32 training points, using a float64 copy of the selected model and a finite-difference step of $10^{-5}$;
- selected-epoch training, validation and test NLLs, data hashes, seeds, configuration and software versions.

The scalar checks are in `summary.json`, and per-point determinant estimates are in `jacobian_checks.npz`. Floating-point round-trip errors need not be exactly zero. Finite differences also depend on step size and local smoothness; a step that crosses a ReLU boundary can disagree with the derivative at the central point. Agreement on these checks validates the evaluated identities, not the accuracy of the learned density for arbitrary data.

Learning curves and generated samples assess fit quality alongside held-out likelihood. A visually plausible sample plot alone does not establish generalisation. The experiment is a synthetic density-estimation exercise; it does not establish performance on physical observations.
