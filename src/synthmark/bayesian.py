"""Training and use of the learned Bayesian detector.

The mean-of-g-values detector treats every position and every depth as equally
informative.  They are not: the tournament sampling procedure leaves a
characteristic, model-specific pattern in how much bias each depth carries.  The
Bayesian detector from the SynthID-Text paper learns that pattern, which buys a
substantial amount of power exactly where the mean detector is weakest -- short
texts.

The cost is that the detector is *fitted*, so it must be trained per
(model, key, sampling-configuration) triple, and it needs a held-out set to be
evaluated honestly.  Training data is cheap to produce: generate watermarked and
unwatermarked text from the model you are protecting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import BayesianDetectorConfig, BayesianDetectorModel

from .detect import Detector


@dataclass
class TrainingRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float


def build_dataset(
    detector: Detector,
    watermarked_texts: Sequence[str],
    unwatermarked_texts: Sequence[str],
    *,
    batch_size: int = 16,
    max_positions: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute padded g-values, masks and labels for a labelled corpus.

    Returns ``(g_values, mask, labels)`` with shapes ``(N, T, depth)``,
    ``(N, T)`` and ``(N,)``.  Sequences shorter than ``T`` are zero-masked, so
    padding contributes nothing to the likelihood.
    """
    chunks_g, chunks_m, labels = [], [], []
    for texts, label in ((watermarked_texts, 1.0), (unwatermarked_texts, 0.0)):
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            g, m = detector.g_values(batch)
            if g.shape[1] == 0:
                continue
            chunks_g.append(g.cpu())
            chunks_m.append(m.cpu().float())
            labels.extend([label] * len(batch))

    if not chunks_g:
        raise ValueError("no scoreable text: every input was shorter than ngram_len")

    max_t = max(c.shape[1] for c in chunks_g)
    if max_positions is not None:
        max_t = min(max_t, max_positions)
    depth = chunks_g[0].shape[2]

    n = sum(c.shape[0] for c in chunks_g)
    G = torch.zeros((n, max_t, depth))
    M = torch.zeros((n, max_t))
    i = 0
    for g, m in zip(chunks_g, chunks_m):
        t = min(g.shape[1], max_t)
        G[i : i + g.shape[0], :t] = g[:, :t]
        M[i : i + g.shape[0], :t] = m[:, :t]
        i += g.shape[0]
    return G, M, torch.tensor(labels, dtype=torch.float32)


def train_bayesian_detector(
    g_values: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    watermarking_depth: int,
    val_fraction: float = 0.2,
    epochs: int = 250,
    lr: float = 1e-3,
    batch_size: int = 64,
    l2_weight: float = 0.0,
    patience: int = 25,
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> tuple[BayesianDetectorModel, list[TrainingRecord]]:
    """Fit the Bayesian detector, selecting the epoch by held-out AUC.

    Model selection uses validation AUC rather than validation loss because AUC
    is what the deployment cares about, and the BCE loss is dominated by the
    regularisation term on ``delta``.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n = g_values.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = max(2, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = torch.device(device)
    Gtr, Mtr, Ytr = g_values[train_idx].to(device), mask[train_idx].to(device), labels[train_idx].to(device)
    Gva, Mva, Yva = g_values[val_idx].to(device), mask[val_idx].to(device), labels[val_idx].to(device)

    config = BayesianDetectorConfig(watermarking_depth=watermarking_depth)
    model = BayesianDetectorModel(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[TrainingRecord] = []
    best_auc, best_state, since_best = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(Gtr.shape[0], generator=g).to(device)
        total, seen = 0.0, 0
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            opt.zero_grad()
            _, loss = model(
                g_values=Gtr[idx], mask=Mtr[idx], labels=Ytr[idx], loss_batch_weight=l2_weight
            )
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
            seen += len(idx)

        model.eval()
        with torch.no_grad():
            probs, vloss = model(g_values=Gva, mask=Mva, labels=Yva, loss_batch_weight=l2_weight)
            vloss = vloss.detach()
        y = Yva.cpu().numpy()
        p = probs.detach().cpu().numpy()
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        history.append(TrainingRecord(epoch, total / max(seen, 1), float(vloss), auc))

        if auc > best_auc:
            best_auc, since_best = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        if verbose and epoch % 25 == 0:
            print(f"  epoch {epoch:4d}  train {total/max(seen,1):.4f}  val {float(vloss):.4f}  val_auc {auc:.4f}")
        if since_best >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best val_auc {best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def save_detector(model: BayesianDetectorModel, path: str | Path, *, model_name: str, watermarking_config: dict) -> Path:
    """Persist the detector together with the config needed to reproduce it.

    The saved ``watermarking_config`` contains the secret key.  Treat the saved
    detector directory with the same care as the key file itself.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.config.set_detector_information(model_name, watermarking_config)
    model.save_pretrained(path)
    return path


def load_detector(path: str | Path, device: str | torch.device = "cpu") -> BayesianDetectorModel:
    model = BayesianDetectorModel.from_pretrained(path).to(torch.device(device))
    model.eval()
    return model
