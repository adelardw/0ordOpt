# SOLUTION

## Reproducibility

```bash
pip install -r requirements.txt
python validate.py --data_dir ./data --batch_size 32 --n_batches 32 --output results.json
```

Defaults to the `irls` head initializer. The selection is exposed via the
`HEAD_SOLVER` env var in `head_init.py` (`irls` | `newton` | `ridge` | `approx`
| `xavier` | `orthogonal` | `small` | `kaiming`). On the first run features
are extracted from CIFAR-100 (hflip TTA, ~150k rows × 512 dims)
and cached at `./data/.features_cache_viewstack.pt`. All subsequent runs reuse
the cache.

Hyper-parameters used by the chosen IRLS solver (in `head_init.py`):
`_RIDGE_LAMBDA = 30.0`, `_LABEL_SMOOTH = 0.1`, `_IRLS_MAX_ITER = 200`,
`_IRLS_TOL = 1e-9`, `_CANONICAL_REPS = 2`, scale `224`.

Reported on MPS (macOS, M4 Pro)

| Checkpoint | Top-1 |
|---|---|
| 1. Baseline (ImageNet head) | 0.37% |
| 2. Initialized head (no FT) | **69.34%** |
| 3. Fine-tuned (ZO, 32×32)   | 69.34% |

## Final approach

Only `head_init.py` is the lever in this submission (the other editable files
keep the skeleton augmentations and a momentum-clipped SPSA-style optimizer).
The head is initialized analytically by fitting a multinomial logistic
regression on frozen ResNet-18 backbone features — no gradients of the
network are computed, and the 8192-sample ZO budget is untouched (feature
extraction happens entirely outside `ZeroOrderOptimizer.step`).

Pipeline:

1. **Feature extraction with hflip TTA.** A single resize scale `224`
   × `{identity, hflip}` gives 2 views per training image. The canonical
   `Resize(224)` view (which is exactly what `validate.py` feeds the model
   at evaluation) is replicated `_CANONICAL_REPS=2` times so that the
   training distribution is centered on the validation distribution while
   still benefiting from flip-augmented diversity. Final design matrix is
   ~150 000 × 512.
2. **Solver: Bohning-bound IRLS** (`solve_irls`). Initialised from the
   regularised one-hot ridge solution and iterated to convergence on the true
   multinomial cross-entropy with L2 penalty (bias unpenalised), label
   smoothing 0.1. The Bohning upper bound on the Hessian gives a closed-form
   Newton-like step at each iteration that monotonically decreases the loss
   and converges to the global CE optimum because the problem is convex.

This gives 69.34% top-1 *without any optimizer steps at all* — the entire
checkpoint-3 number on this submission comes from the head init.

## Why this choice — comparison of all `head_init` variants

The same `validate.py --batch_size 32 --n_batches 32` was run for each
candidate solver. Init-head accuracy (checkpoint 2) isolates head quality;
fine-tuned (checkpoint 3) shows that the current ZO step does not move the
needle from a strong init at this budget, so head quality dominates.

| Solver | init_head | finetuned |
|---|---:|---:|
| `xavier` (random) | 1.21% | 1.21% |
| `orthogonal` (random) | 0.90% | 0.90% |
| `small` (xavier × 0.01) | 1.21% | 1.21% |
| `approx` — normal eqs on smoothed logit targets, no L2 | 61.65% | 61.65% |
| `ridge` — Taylor-expanded sigmoid + L2 (`λ=30`) | 61.67% | 61.66% |
| `newton` — one exact Newton step from `ridge`, per-class diag Hessian | 61.85% | 61.84% |
| **`irls`** — Bohning-bound IRLS to convergence | **69.34%** | **69.34%** |

Observations:

- **Random inits give ≈chance (1%).** Expected — the new 100-class head is
  uncorrelated with the 1000-class ImageNet head.
- **`approx`, `ridge`, and `newton` are competitive with each other** (61.65–61.85%), all extracting most of the linear-probe signal from the feature matrix. `irls` is in a separate tier, ~7.5 points above them.
- **A single Newton step** from the ridge solution buys +0.18% by replacing
  the global curvature constant `σ'(0)=0.25` with the actual per-class
  diagonal Hessian — improvement is small because the ridge point is already
  near the optimum on this objective.
- **IRLS wins by ~7.5 points** because it iterates the Bohning step to true
  CE convergence: the closed-form `ridge`/`approx` solvers are first-order
  Taylor approximations of softmax-CE around `Xw=0`, so they bias the head
  toward a low-confidence regime; IRLS removes that bias.

## Experiments and discarded ideas

- **Random initializations from the README hint list** (`xavier`,
  `orthogonal`, scaled-small). All collapse to chance — kept as a sanity
  baseline only.
- **Approx (normal eqs, no L2)** — works because the system is overdetermined
  (n=150k ≫ d=513), marginally trails ridge (61.65% vs 61.67%) — noise-level
  difference. Discarded.
- **Ridge (Taylor sigmoid + L2)** — clean closed form, almost identical to
  `approx` here; the `λ=30` choice barely matters at this n. Used as warm
  start for `newton`/`irls` but not as the final solver.
- **One-shot Newton from ridge** — faster than IRLS (one batched LU solve),
  but only +0.18% over ridge (61.85% vs 61.67%) because the surrogate is
  already near the CE optimum. Discarded.
- **Tweaking the ZO optimizer** is not part of this submission's gain — at
  the 32×32 budget the fine-tuned and init-head accuracies are equal, so
  any improvement here would be additive. Kept the skeleton SPSA-style
  estimator with momentum and gradient clipping.
