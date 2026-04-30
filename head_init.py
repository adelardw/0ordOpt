"""
head_init.py — Linear-probe head initialization with hflip-TTA features.

Extracts ResNet18 features on the CIFAR100 train set with horizontal-flip
test-time augmentation (averaged over the original and flipped views) and
fits a multinomial logistic-regression classifier on full batch via L-BFGS.
The result is written directly into the new 100-class head, so checkpoint 2
already reflects this linear probe (no ZO budget is consumed here).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models


_DATA_DIR = "./data"
_CACHE_PATH = os.path.join(_DATA_DIR, ".features_cache_viewstack.pt")
_BATCH_SIZE = 128
_LOGREG_WD = 1e-4
_LOGREG_MAX_ITER = 300
_LABEL_SMOOTH = 0.1
_CANONICAL_REPS = 2   # replicate the canonical Resize(224) view to keep it dominant


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _backbone() -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval()
    return m


def _extract_features() -> tuple[torch.Tensor, torch.Tensor]:
    """Stack each (scale × hflip) view as a separate training sample.

    Six views per image: {Resize(224), Resize(240)→CC(224), Resize(256)→CC(224)}
    × {no flip, hflip}. The canonical Resize(224) views are replicated
    ``_CANONICAL_REPS`` times so the val distribution stays dominant.
    Returns X of shape (N·V_eff, 512) and y of shape (N·V_eff,).
    """
    if os.path.exists(_CACHE_PATH):
        blob = torch.load(_CACHE_PATH, map_location="cpu")
        return blob["X"], blob["y"]

    import torchvision.transforms as T
    _MEAN = (0.5071, 0.4867, 0.4408)
    _STD  = (0.2675, 0.2565, 0.2761)

    def _mk(resize: int) -> T.Compose:
        if resize == 224:
            return T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(_MEAN, _STD)])
        return T.Compose([T.Resize(resize), T.CenterCrop(224),
                          T.ToTensor(), T.Normalize(_MEAN, _STD)])

    scales = (224, 240, 256)

    device = _pick_device()
    backbone = _backbone().to(device)

    # Per-view containers: (scale, flip) -> list of (B, 512) tensors.
    view_feats: dict[tuple[int, bool], list[torch.Tensor]] = {}
    for s in scales:
        for flip in (False, True):
            view_feats[(s, flip)] = []
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
    for (s, flip), chunks in view_feats.items():
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


def _solve_logreg(X: torch.Tensor, y: torch.Tensor, num_classes: int,
                  wd: float, max_iter: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Multinomial logistic regression via full-batch L-BFGS with strong-Wolfe."""
    device = _pick_device()
    Xd = X.to(device)
    yd = y.long().to(device)

    d = Xd.shape[1]
    W = torch.zeros(num_classes, d, device=device, requires_grad=True)
    b = torch.zeros(num_classes, device=device, requires_grad=True)

    opt = torch.optim.LBFGS(
        [W, b], lr=1.0, max_iter=max_iter, history_size=20,
        tolerance_grad=1e-7, tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        logits = Xd @ W.t() + b
        loss = torch.nn.functional.cross_entropy(
            logits, yd, label_smoothing=_LABEL_SMOOTH,
        )
        loss = loss + 0.5 * wd * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().cpu(), b.detach().cpu()


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the 100-class head via logreg on hflip-TTA ResNet18 features."""
    num_classes, in_features = layer.weight.shape
    try:
        X, y = _extract_features()
        assert X.shape[1] == in_features
        W, b = _solve_logreg(X, y, num_classes, _LOGREG_WD, _LOGREG_MAX_ITER)
        with torch.no_grad():
            layer.weight.copy_(W.to(layer.weight.dtype))
            layer.bias.copy_(b.to(layer.bias.dtype))
    except Exception as e:
        print(f"[head_init] linear probe failed ({e}); falling back to xavier.")
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
