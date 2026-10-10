# References

1. Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio. *Density estimation using Real NVP*. ICLR 2017. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803). Affine coupling transformations provide invertible mappings with tractable Jacobian determinants.
2. Simon J. D. Prince. *Understanding Deep Learning*. MIT Press, 2023, Chapter 16: Normalizing flows. [Author's book website](https://udlbook.github.io/udlbook/). This treatment provides the normalising-flow formulation.
3. M. Cranmer. *Deep Learning*: specification for the architecture, diagnostics, shear composition and FLOP-counting tasks.

## Software

- [PyTorch](https://pytorch.org/) — tensor computation, automatic differentiation, neural networks and optimisation.
- [NumPy](https://numpy.org/) — numerical arrays and random sampling.
- [Matplotlib](https://matplotlib.org/) — scientific figures.

The project implements affine coupling directly in PyTorch; it does not depend on a normalising-flow library.
