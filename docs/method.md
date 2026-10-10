# Methods

## Affine coupling flow

The model learns a continuous density over two-dimensional coordinates. A standard Gaussian latent variable $z\sim\mathcal{N}(0,I)$ is transformed by a sequence of invertible affine coupling layers. The default model has six layers, with masks alternating between `(1, 0)` and `(0, 1)`.

For a binary mask $m$, a shared network takes $m\odot h$ as input:

$$
\operatorname{Linear}(D,H)\to\operatorname{ReLU}\to\operatorname{Linear}(H,2D).
$$

Its outputs define the scale and translation,

$$
s=(1-m)\odot\tanh(s_{\mathrm{raw}}),\qquad
 t=(1-m)\odot t_{\mathrm{raw}}.
$$

The forward and inverse maps are

$$
h'=m\odot h+(1-m)\odot(h\odot e^s+t),
$$

$$
h=m\odot h'+(1-m)\odot((h'-t)\odot e^{-s}).
$$

Both maps evaluate $s$ and $t$ from the unchanged coordinates. The Jacobian is triangular after grouping fixed and transformed features. Its forward log absolute determinant is $\sum_j s_j$; the inverse contributes $-\sum_j s_j$. A coupling layer generally changes volume: its determinant is $\exp(\sum_j s_j)$, not necessarily one. The `tanh` bound keeps each transformed scale factor between $e^{-1}$ and $e$ in exact arithmetic.

For the complete sampling map $f$, density evaluation runs the layers in reverse:

$$
\log p_X(x)
=-\frac12\left(\|f^{-1}(x)\|^2+D\log(2\pi)\right)
+\log\left|\det J_{f^{-1}}(x)\right|.
$$

The inverse log-determinant is accumulated across layers. Sampling instead draws a Gaussian latent point and applies the forward maps. The model implements affine coupling directly in PyTorch; see [References](references.md) for the method and teaching sources.

## Data and training

The default example uses generated two-moons coordinates with Gaussian noise of standard deviation 0.08. The 800/100/100 training, validation and test splits use separate random streams with data seed 42. Labels are for visualisation only. The geometry, seed handling and external CSV format are documented in [Data](data.md).

The default configuration is:

| Setting | Value |
|---|---:|
| Coordinates $D$ | 2 |
| Coupling layers $K$ | 6 |
| Hidden width $H$ | 64 |
| Optimiser | Full-batch Adam |
| Initial learning rate | 0.003 |
| Learning-rate schedule | Cosine annealing to zero over 4,000 epochs |
| Maximum epochs | 4,000 |
| Early-stopping patience | 300 epochs |
| Model seed | 42 |
| Training precision and device | float32, CPU |

One epoch is one full-batch optimiser update. After every update, the code evaluates both training and validation negative log likelihood (NLL) on the same model state. The logged learning rate is the rate used for that update. The validation minimum determines the saved weights; training stops after 300 consecutive epochs without a strictly lower validation NLL, or at the epoch limit. The lowest-validation-NLL weights are restored before final evaluation.

The test NLL is evaluated once after fitting and model selection. Test data are not passed to the training function. The example learning rate was selected by validation NLL from 0.0005, 0.001, 0.0013862987370816613 and 0.003, with the architecture, seeds, schedule and patience held fixed. The test scores were not used to choose the learning rate. These settings are defaults for this synthetic example, not a claim of globally optimal hyperparameters. Training without validation is available for a small-subset diagnostic; this requires disabling patience and retains the final epoch.

NLL is the mean negative log density, measured in **nats per point**. It may be negative because a continuous probability density can exceed one. It is not a classification loss or accuracy measure, and its numerical value depends on the coordinate convention and data distribution.

## Shear family

A deterministic shear is appended to the trained sampling map:

$$
g_\alpha(x_1,x_2)=(x_1+\alpha x_2,x_2),\qquad
 g_\alpha^{-1}(y_1,y_2)=(y_1-\alpha y_2,y_2).
$$

Its Jacobian is the upper-triangular matrix

$$
J_{g_\alpha}=\begin{pmatrix}1&\alpha\\0&1\end{pmatrix},
\qquad \det J_{g_\alpha}=1.
$$

Thus $f_\alpha=g_\alpha\circ f$ defines a valid density without retraining:

$$
\log p_\alpha(y)=\log p_X(g_\alpha^{-1}(y)).
$$

Both directions of the shear have zero log-determinant. The figure compares $\alpha\in\{-1,0,1\}$ using common base samples so that changes reflect the transformation rather than sampling variation.

## Idealised operation count

The operation count describes one density evaluation for a batch of $B$ points under a fixed mathematical convention. A dense `Linear(a, b)` layer costs $2ab$ operations per point, including bias additions. Each scalar `exp`, `log` or `tanh` is assigned ten operations, each ReLU one operation, and Gaussian base-density evaluation twenty per point. Mask sparsity is not exploited.

| Calculation per coupling layer | Operations per point |
|---|---:|
| Input mask | $D$ |
| Two dense linear layers | $6DH$ |
| ReLU | $H$ |
| `tanh` | $10D$ |
| Scale and translation masks | $2D$ |
| Inverse affine map | $16D$ |
| Inverse log-determinant | $D$ |
| **Layer total** | **$6DH+H+30D$** |

One additional operation per layer accumulates its log-determinant. The Gaussian base density and final addition contribute another 21 operations per point:

$$
\mathrm{FLOPs}=B\left[K(6DH+H+30D+1)+21\right].
$$

For $D=2$, $K=6$, $H=64$ and $B=800$, this gives **4,303,200** operations. Mask-complement construction, Python and indexing overhead, data movement, optimiser updates and backward differentiation are excluded. This is an idealised count, not a measurement of hardware instructions, runtime or training cost.
