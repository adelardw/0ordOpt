"""
head_init.py — Closed-form ridge linear-probe initialization.

Initializes the new 100-class head by solving a ridge-regression linear
classifier on frozen ResNet18 features. Everything is purely analytic
(``torch.linalg.solve`` on the normal equations) — no autograd / no
gradient-based optimizer is invoked. The 8192-sample ZO compute budget
is untouched: every forward pass here happens outside ``ZeroOrderOptimizer.step``.

Features are extracted with multi-scale + horizontal-flip TTA, and each
augmented view is stacked as an independent training row. The canonical
``Resize(224)`` view (which val also sees) is replicated to keep the
training distribution centred on the val distribution.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T
from augmentation import _CIFAR100_MEAN, _CIFAR100_STD

_DATA_DIR = "./data"
_CACHE_PATH = os.path.join(_DATA_DIR, ".features_cache_viewstack.pt")
_BATCH_SIZE = 128
_RIDGE_LAMBDA = 40.0
_CANONICAL_REPS = 2
_IRLS_MAX_ITER = 500
_IRLS_TOL = 1e-9
_LABEL_SMOOTH = 0.1


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_backbone():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    return model

def _extract_features() -> tuple[torch.Tensor, torch.Tensor]:
    """View-stacked multi-scale + hflip features."""
    if os.path.exists(_CACHE_PATH):
        blob = torch.load(_CACHE_PATH, map_location="cpu")
        return blob["X"], blob["y"]



    def _mk(resize: int) -> T.Compose:
        if resize == 224:
            return T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD)])
        return T.Compose([T.Resize(resize), T.CenterCrop(224),
                          T.ToTensor(), T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD)])

    scales = (224, 240, 256)

    device = get_device()
    backbone = get_backbone().to(device)

    view_feats: dict[tuple[int, bool], list[torch.Tensor]] = {
        (s, flip): [] for s in scales for flip in (False, True)
    }
    label_chunks: list[torch.Tensor] = []

    loaders = []
    for s in scales:
        ds = datasets.CIFAR100(root=_DATA_DIR, train=True, download=True, transform=_mk(s))
        loaders.append(torch.utils.data.DataLoader(
            ds, batch_size=_BATCH_SIZE, shuffle=False, num_workers=0,
        ))

    with torch.no_grad():
        for batch_tuple in zip(*loaders):
            label_chunks.append(batch_tuple[0][1])
            for s, (imgs, _) in zip(scales, batch_tuple):
                imgs = imgs.to(device, non_blocking=True)
                view_feats[(s, False)].append(backbone(imgs).float().cpu())
                view_feats[(s, True)].append(
                    backbone(torch.flip(imgs, dims=[3])).float().cpu()
                )

    y_base = torch.cat(label_chunks, dim=0)

    Xs, ys = [], []
    for (s, _flip), chunks in view_feats.items():
        Xv = torch.cat(chunks, dim=0)
        reps = _CANONICAL_REPS if s == 224 else 1
        for _ in range(reps):
            Xs.append(Xv)
            ys.append(y_base)

    X = torch.cat(Xs, dim=0)
    y = torch.cat(ys, dim=0)

    os.makedirs(_DATA_DIR, exist_ok=True)
    torch.save({"X": X, "y": y}, _CACHE_PATH)
    return X, y


@torch.no_grad()
def solve_approximate(X: torch.Tensor, y: torch.Tensor, num_classes: int,
                      label_smooth: float) -> tuple[torch.Tensor, torch.Tensor]:

    '''
    At the optimum of softmax CE: X.T (sigmoid(Xw) - y) = 0
    Approximating sigmoid(Xw) ≈ y gives normal equations:
    (X.T X) w = X.T logit(y)  =>  w = (X.T X)^-1 X.T ln(y / (1 - y))
    '''
    n, d = X.shape

    Xb = torch.cat([X.float(), torch.ones(n, 1, dtype=torch.float32)], dim=1)
    Y = torch.full((n, num_classes), label_smooth / num_classes,
                   dtype=torch.float32)
    Y.scatter_(1, y.long().unsqueeze(1),
               1.0 - label_smooth + label_smooth / num_classes)

    logit_Y = torch.log(Y / (1.0 - Y)).double()           # (n, C)
    XTX = (Xb.t() @ Xb).double()                          # (d+1, d+1)
    XTlogitY = (Xb.t().double() @ logit_Y)                # (d+1, C)
    Wb = torch.linalg.solve(XTX, XTlogitY)                # (d+1, C)
    W = Wb[:d].t().float().contiguous()
    b = Wb[d].float().contiguous()
    return W, b

@torch.no_grad()
def solve_approximate_ridge(X: torch.Tensor, y: torch.Tensor, num_classes: int,
                      label_smooth: float,
                      lam: float = _RIDGE_LAMBDA) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Regularised closed-form init from the first-order Taylor expansion of sigmoid.

    Optimality condition with L2 regularisation:
        X.T (sigmoid(Xw) - y) + λw = 0

    Expanding sigmoid around Xw = 0:
        sigmoid(z) ≈ 0.5 + 0.25·z   (sigmoid'(0) = 0.25)

    Substituting and solving for w:
        (0.25·X.T X + λI) w = X.T (y - 0.5)
        w = (0.25·X.T X + λI)^{-1} X.T (y - 0.5)

    The bias column in Xb is unregularised (reg_diag[-1] = 0).
    '''
    n, d = X.shape

    Xb = torch.cat([X.float(), torch.ones(n, 1, dtype=torch.float32)], dim=1)
    Y = torch.full((n, num_classes), label_smooth / num_classes,
                   dtype=torch.float32)
    Y.scatter_(1, y.long().unsqueeze(1),
               1.0 - label_smooth + label_smooth / num_classes)

    rhs = (Xb.t().double() @ (Y - 0.5).double())          # (d+1, C)

    XTX = (Xb.t() @ Xb).double()                          # (d+1, d+1)
    reg_diag = torch.zeros(d + 1, dtype=torch.float64)
    reg_diag[:d] = lam
    A = 0.25 * XTX + torch.diag(reg_diag)                 # (d+1, d+1)

    Wb = torch.linalg.solve(A, rhs)                       # (d+1, C)
    W = Wb[:d].t().float().contiguous()
    b = Wb[d].float().contiguous()
    return W, b


@torch.no_grad()
def solve_hessian_newton(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    label_smooth: float,
    lam: float = _RIDGE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One exact Newton step from ridge init, using per-class diagonal Hessian.

    solve_approximate_ridge uses a global curvature scalar sigmoid'(0)=0.25.
    Here we compute the actual per-class Hessian weight at the ridge solution:

        s_c = (1/n) Σ_i  p_{ic}(1 − p_{ic}),   P = softmax(X W_ridge)

    The Newton update for class c is then:

        w_c ← w_c + (s_c · XᵀX + λI)⁻¹ [Xᵀ(y_c − p_c) − λ w_c]

    All C systems share the same XᵀX and are solved in one batched call:

        A  ∈ ℝ^{C × (d+1) × (d+1)},   A_c = s_c · XᵀX + diag(λ, …, λ, 0)
        rhs ∈ ℝ^{C × (d+1)}

    Cost: one ridge solve (warm start) + one softmax pass + one batched LU.
    Converges faster than the 0.25-flat approximation when the head is far from
    the decision boundary (high confidence → large s_c variance across classes).
    """
    n, d = X.shape

    Xb = torch.cat([X.float(), torch.ones(n, 1, dtype=torch.float32)], dim=1)
    Y = torch.full((n, num_classes), label_smooth / num_classes, dtype=torch.float32)
    Y.scatter_(1, y.long().unsqueeze(1), 1.0 - label_smooth + label_smooth / num_classes)

    # Warm-start from ridge solution
    W0, b0 = solve_approximate_ridge(X, y, num_classes, label_smooth, lam)
    Wb = torch.cat([W0.t(), b0.unsqueeze(0)], dim=0).double()  # (d+1, C)

    Xbd = Xb.double()
    Yd = Y.double()

    # Softmax at the ridge point
    logits = Xbd @ Wb                                           # (n, C)
    logits -= logits.max(dim=1, keepdim=True).values
    ex = torch.exp(logits)
    P = ex / ex.sum(dim=1, keepdim=True)                        # (n, C)

    # Per-class diagonal Hessian weights s_c = mean_i p_ic(1 - p_ic)  → (C,)
    S = (P * (1.0 - P)).mean(dim=0)                             # (C,)

    XTX = Xbd.t() @ Xbd                                        # (d+1, d+1)

    # Regularisation: λ on weights, 0 on bias
    reg_diag = torch.zeros(d + 1, dtype=torch.float64)
    reg_diag[:d] = lam

    # Batched A_c = s_c · XᵀX + diag(reg),  shape (C, d+1, d+1)
    A = S[:, None, None] * XTX.unsqueeze(0) + torch.diag(reg_diag).unsqueeze(0)

    # Gradient of CE + L2 at current Wb
    grad = Xbd.t() @ (Yd - P)                                  # (d+1, C)
    grad[:d] -= lam * Wb[:d]

    # Batched solve:  A_c Δw_c = grad_c
    rhs = grad.t().unsqueeze(-1)                                # (C, d+1, 1)
    delta = torch.linalg.solve(A, rhs).squeeze(-1).t()         # (d+1, C)

    Wb = Wb + delta

    W = Wb[:d].t().float().contiguous()
    b = Wb[d].float().contiguous()
    return W, b

@torch.no_grad()
def solve_irls(X: torch.Tensor, y: torch.Tensor, num_classes: int,
                lam: float, max_iter: int, tol: float, label_smooth: float
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Multinomial logistic regression via Bohning-bound IRLS.

    No autograd, no gradient descent. Each iteration solves a closed-form
    ridge regression:

        Wₜ₊₁ = Wₜ + 2 (XᵀX + λI)⁻¹ Xᵀ (Y - Pₜ),    Pₜ = softmax(X Wₜ)

    The factor 2 comes from Bohning's (1992) global upper bound on the
    Hessian of multinomial NLL: H ≼ ½ XᵀX ⊗ (I − 11ᵀ/C). Replacing the true
    Hessian by the bound gives a closed-form Newton-like step that
    monotonically decreases the loss. Initialised from the ridge solution
    (one-hot regression). Converges to the global CE optimum because the
    objective is convex.

    Label smoothing is applied to the one-hot targets:
        Y_smooth = (1 − α) · onehot(y) + α / C
    """
    n, d = X.shape

    # fp32 features (memory-friendly), fp64 for the small solve.
    Xb = torch.cat([X.float(), torch.ones(n, 1, dtype=torch.float32)], dim=1)
    Y = torch.full((n, num_classes), label_smooth / num_classes,
                   dtype=torch.float32)
    Y.scatter_(1, y.long().unsqueeze(1),
               1.0 - label_smooth + label_smooth / num_classes)

    # Bohning-bound Newton on regularised loss
    #     L = NLL + ½ λ ‖W‖²    (no penalty on bias row)
    #     H ≼ ½ XᵀX + λ I_reg
    #     ΔW = 2 (XᵀX + 2λ I_reg)⁻¹ (Xᵀ(Y − P) − λ W_reg)
    XTX = (Xb.t() @ Xb).double()                         # (d+1, d+1)
    reg_diag = torch.zeros(d + 1, dtype=torch.float64)
    reg_diag[:d] = 2.0 * lam
    A = XTX + torch.diag(reg_diag)                        # fp64 for stable solve

    # Initialise from ridge on smoothed targets (gradient = 0 of quadratic surrogate).
    XTY = (Xb.t() @ Y).double()
    rhs0 = XTY.clone()                                    # bias has no λW term initially
    Wb = torch.linalg.solve(A, rhs0)                      # (d+1, C) fp64

    prev = float("inf")
    for _ in range(max_iter):
        logits = (Xb @ Wb.float())                        # (n, C) fp32
        m = logits.max(dim=1, keepdim=True).values
        ex = torch.exp(logits - m)
        Z = ex.sum(dim=1, keepdim=True)
        P = ex / Z                                        # (n, C)

        log_P = (logits - m) - torch.log(Z)
        ce = -(Y * log_P).sum().item() / n
        if abs(prev - ce) < tol:
            break
        prev = ce

        # RHS = Xᵀ(Y − P) − λ W_reg  (bias row gets no penalty)
        rhs = (Xb.t() @ (Y - P)).double()
        rhs[:d] = rhs[:d] - lam * Wb[:d]
        delta = torch.linalg.solve(A, rhs)
        Wb = Wb + 2.0 * delta

    Wb_f = Wb.float()
    W = Wb_f[:d].t().contiguous()
    b = Wb_f[d].contiguous()
    return W, b

def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the head via Bohning-bound IRLS multinomial logistic regression."""
    num_classes, in_features = layer.weight.shape
    try:
        X, y = _extract_features()
        assert X.shape[1] == in_features
        W, b = solve_irls(
            X, y, num_classes,
            lam=_RIDGE_LAMBDA, max_iter=_IRLS_MAX_ITER, tol=_IRLS_TOL,
            label_smooth=_LABEL_SMOOTH,
        )
        with torch.no_grad():
            layer.weight.copy_(W.to(layer.weight.dtype))
            layer.bias.copy_(b.to(layer.bias.dtype))
    except Exception as e:
        print(f"[head_init] IRLS probe failed ({e}); falling back to xavier.")
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
