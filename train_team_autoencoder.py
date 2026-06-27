"""Autoencoder nas equipas (reunião 2026-05-15 — latent space do orientador)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_move_count_predictor import pick_device


class TeamAutoEncoder(nn.Module):
    def __init__(self, team_dim: int, latent_dim: int, hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        enc: list[nn.Module] = []
        d = team_dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        enc.append(nn.Linear(d, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        d = latent_dim
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec.append(nn.Linear(d, team_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def extract_team_matrix(X: np.ndarray, team_dim: int) -> np.ndarray:
    if X.shape[1] < team_dim:
        raise ValueError(f"X cols {X.shape[1]} < team_dim {team_dim}")
    return np.asarray(X[:, :team_dim], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="team_autoencoder.pt")
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden_dims", type=str, default="512,256")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    d = np.load(args.data, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    team_dim = int(np.asarray(d["team_encoder_dim"]).reshape(())) * 2
    if team_dim <= 0:
        team_dim = X.shape[1] - int(d["rule_params"].shape[1]) if "rule_params" in d.files else X.shape[1]

    teams = extract_team_matrix(X, team_dim)
    n = teams.shape[0]
    perm = np.random.permutation(n)
    n_val = int(n * args.val_fraction)
    tr, va = perm[n_val:], perm[:n_val]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(teams[tr])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(teams[va])),
        batch_size=args.batch_size,
        shuffle=False,
    )

    hidden = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    model = TeamAutoEncoder(team_dim, int(args.latent_dim), hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        nb = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
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
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, _ = model(xb)
                va_loss += float(loss_fn(recon, xb).item())
                nb += 1
        va_loss /= max(nb, 1)
        print(f"epoch {epoch}/{args.epochs}  train_mse={tr_loss:.5f}  val_mse={va_loss:.5f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    out = Path(args.out)
    payload = {
        "state_dict": model.state_dict(),
        "team_dim": team_dim,
        "latent_dim": int(args.latent_dim),
        "hidden_dims": hidden,
        "best_val_recon": best_val,
        "data_path": str(Path(args.data).resolve()),
    }
    torch.save(payload, out)
    out.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2),
        encoding="utf-8",
    )
    print(f"Guardado: {out.resolve()}  best_val_recon={best_val:.5f}")


if __name__ == "__main__":
    main()
