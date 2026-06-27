"""Previsor da ordem agregada de moves (sequência por turno, lado 0)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_move_count_predictor import pick_device


class SeqPredictorMLP(nn.Module):
    def __init__(self, in_dim: int, max_len: int, n_classes: int, hidden: tuple[int, ...]):
        super().__init__()
        self.max_len = max_len
        self.n_classes = n_classes
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, max_len * n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.max_len, self.n_classes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="rollout_seq_predictor.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dims", type=str, default="512,256")
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    d = np.load(args.data, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y_seq"], dtype=np.int64)
    max_len = int(np.asarray(d["max_len"]).reshape(()))
    n_classes = int(np.asarray(d["n_classes"]).reshape(()))

    n = X.shape[0]
    perm = np.random.permutation(n)
    n_val = int(n * args.val_fraction)
    tr, va = perm[n_val:], perm[:n_val]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[va]), torch.from_numpy(Y[va])),
        batch_size=args.batch_size,
        shuffle=False,
    )

    hidden = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    model = SeqPredictorMLP(X.shape[1], max_len, n_classes, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        nb = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, n_classes), yb.view(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            nb += 1
        tr_loss /= max(nb, 1)

        model.eval()
        va_loss = 0.0
        nb = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                va_loss += float(F.cross_entropy(logits.view(-1, n_classes), yb.view(-1), ignore_index=-100).item())
                nb += 1
        va_loss /= max(nb, 1)
        print(f"epoch {epoch}/{args.epochs}  train_ce={tr_loss:.5f}  val_ce={va_loss:.5f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    out = Path(args.out)
    payload = {
        "state_dict": model.state_dict(),
        "in_dim": int(X.shape[1]),
        "max_len": max_len,
        "n_classes": n_classes,
        "hidden_dims": hidden,
        "best_val_ce": best_val,
        "data_path": str(Path(args.data).resolve()),
    }
    torch.save(payload, out)
    out.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2),
        encoding="utf-8",
    )
    print(f"Guardado: {out.resolve()}  best_val_ce={best_val:.5f}")


if __name__ == "__main__":
    main()
