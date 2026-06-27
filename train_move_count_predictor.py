"""Treino do previsor de distribuição de uso de moves (agregado por batalha).

Dataset: `move_count_sup_big.npz` (`build_move_count_supervision.py`)
  - Augmentação (reunião 2026-05-15 / notas orientador):
      X,Y -> Xm   (T0||T1, target = contagens T0)
      Y,X -> Ym   (T1||T0, target = contagens T1)
  - Y shape (N, 9) = [2 pkm x 4 moves] + [1 switch global]

Losses (reunião 2026-05-15):
  - L_global: SmoothL1 sobre os 9 alvos
  - L_local:  L_pkm0 + L_pkm1  (CE com distribuição suave por 4 moves)
  - L_switch: regressão ou BCE no canal switch
  - total (local): L_local + w_switch * L_switch
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from move_predictor_config import N_TARGETS


def pick_device(name: str) -> torch.device:
    key = (name or "auto").strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class MLPGlobal(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int = N_TARGETS):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPLocalHeads(nn.Module):
    """Fig. (b) direct MLP com cabeças locais: 4 moves/pkm + 1 switch global."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        trunk: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            trunk.append(nn.Linear(d, h))
            trunk.append(nn.ReLU())
            d = h
        self.trunk = nn.Sequential(*trunk)
        self.head_p0 = nn.Linear(d, 4)
        self.head_p1 = nn.Linear(d, 4)
        self.head_sw = nn.Linear(d, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        return self.head_p0(z), self.head_p1(z), self.head_sw(z)


def counts_to_move_dist(
    counts_4: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    c = counts_4.clamp(min=0.0)
    s = c.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(c, 0.25)
    out = c / (s + eps)
    mask_zero = (s.squeeze(-1) < eps)
    if mask_zero.any():
        out = torch.where(mask_zero.unsqueeze(-1), uniform, out)
    return out


@dataclass
class LossMetrics:
    L_global: float = 0.0
    L_pkm0: float = 0.0
    L_pkm1: float = 0.0
    L_local: float = 0.0
    L_switch: float = 0.0
    L_total: float = 0.0

    def format_line(self, mode: str) -> str:
        if mode == "global":
            return f"L_global={self.L_global:.5f}  L_total={self.L_total:.5f}"
        return (
            f"L_pkm0={self.L_pkm0:.5f}  L_pkm1={self.L_pkm1:.5f}  L_local={self.L_local:.5f}  "
            f"L_switch={self.L_switch:.5f}  L_total={self.L_total:.5f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Treino previsor de contagens / distribuição de moves.")
    ap.add_argument("--data", type=str, required=True, help="move_count_sup_big.npz")
    ap.add_argument("--out", type=str, default="move_count_predictor.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dims", type=str, default="512,512,256")
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--loss",
        choices=("global", "local"),
        default="local",
        help="global=L_global nos 9; local=L_pkm0+L_pkm1+L_switch.",
    )
    ap.add_argument(
        "--switch_loss",
        choices=("mse", "bce_any"),
        default="mse",
        help="L_switch: regressão da contagem ou BCE(usou>=1).",
    )
    ap.add_argument("--w_switch", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    d = np.load(args.data, allow_pickle=True)
    if "X" not in d.files or "Y" not in d.files:
        raise SystemExit("Ficheiro precisa de X e Y (move_count_supervision).")
    X_np = np.asarray(d["X"], dtype=np.float32)
    Y_np = np.asarray(d["Y"], dtype=np.float32)
    if Y_np.shape[1] != N_TARGETS:
        raise SystemExit(f"Esperado Y com {N_TARGETS} colunas, tem {Y_np.shape[1]}")

    n = X_np.shape[0]
    perm = np.random.permutation(n)
    n_val = int(n * args.val_fraction)
    idx_val, idx_tr = perm[:n_val], perm[n_val:]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_np[idx_tr]), torch.from_numpy(Y_np[idx_tr])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_np[idx_val]), torch.from_numpy(Y_np[idx_val])),
        batch_size=args.batch_size,
        shuffle=False,
    )

    hidden = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    in_dim = X_np.shape[1]

    print(
        f"treino: loss={args.loss}  samples={n}  in_dim={in_dim}  "
        f"augment=[Xm,Ym]  targets=[2,4,+1switch]={N_TARGETS}"
    )

    if args.loss == "global":
        model = MLPGlobal(in_dim, hidden, N_TARGETS).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        def forward_loss(xb: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, LossMetrics]:
            pred = model(xb)
            lg = F.smooth_l1_loss(pred, yb, beta=1.0)
            m = LossMetrics(L_global=float(lg.item()), L_total=float(lg.item()))
            return lg, m

    else:
        model = MLPLocalHeads(in_dim, hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        def forward_loss(xb: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, LossMetrics]:
            y0, y1, ysw = yb[:, 0:4], yb[:, 4:8], yb[:, 8:9]
            p0, p1, psw = model(xb)
            t0 = counts_to_move_dist(y0)
            t1 = counts_to_move_dist(y1)
            lp0 = -(t0 * F.log_softmax(p0, dim=-1)).sum(dim=-1).mean()
            lp1 = -(t1 * F.log_softmax(p1, dim=-1)).sum(dim=-1).mean()
            if args.switch_loss == "mse":
                lsw = F.smooth_l1_loss(psw, ysw, beta=1.0)
            else:
                lsw = F.binary_cross_entropy_with_logits(psw, (ysw > 0.5).float())
            total = lp0 + lp1 + args.w_switch * lsw
            m = LossMetrics(
                L_pkm0=float(lp0.item()),
                L_pkm1=float(lp1.item()),
                L_local=float((lp0 + lp1).item()),
                L_switch=float(lsw.item()),
                L_total=float(total.item()),
            )
            return total, m

    def run_epoch(loader: DataLoader, train: bool) -> LossMetrics:
        model.train(train)
        acc = LossMetrics()
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss_t, m = forward_loss(xb, yb)
            if train:
                opt.zero_grad()
                loss_t.backward()
                opt.step()
            acc.L_global += m.L_global
            acc.L_pkm0 += m.L_pkm0
            acc.L_pkm1 += m.L_pkm1
            acc.L_local += m.L_local
            acc.L_switch += m.L_switch
            acc.L_total += m.L_total
            n_batches += 1
        if n_batches:
            for field in ("L_global", "L_pkm0", "L_pkm1", "L_local", "L_switch", "L_total"):
                setattr(acc, field, getattr(acc, field) / n_batches)
        return acc

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, True)
        with torch.inference_mode():
            va = run_epoch(val_loader, False)
        print(
            f"epoch {epoch}/{args.epochs}  train: {tr.format_line(args.loss)}  "
            f"val: {va.format_line(args.loss)}"
        )
        if va.L_total < best_val:
            best_val = va.L_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path(args.out)
    payload = {
        "model_type": args.loss,
        "in_dim": in_dim,
        "hidden_dims": hidden,
        "state_dict": model.state_dict(),
        "best_val_L_total": best_val,
        "data_path": str(Path(args.data).resolve()),
        "switch_loss": args.switch_loss if args.loss == "local" else None,
        "w_switch": float(args.w_switch),
        "n_pokemon": 2,
        "n_moves_per_pokemon": 4,
        "move_count_dim": N_TARGETS,
        "augment": "X,Y->Xm; Y,X->Ym",
        "layout": "Y[0:4] pkm0, Y[4:8] pkm1, Y[8] switch",
    }
    torch.save(payload, out_path)
    print(f"Guardado: {out_path.resolve()}  best_val_L_total={best_val:.5f}")
    out_path.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
