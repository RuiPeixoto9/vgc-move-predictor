"""Previsor de moves no espaço latente (encoder congelado do team autoencoder)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from move_predictor_config import N_TARGETS
from train_move_count_predictor import (
    MLPLocalHeads,
    LossMetrics,
    counts_to_move_dist,
    pick_device,
)
from train_team_autoencoder import TeamAutoEncoder, extract_team_matrix


def load_frozen_encoder(ae_ckpt: str | Path, device: torch.device) -> tuple[TeamAutoEncoder, int, int, int]:
    ck = torch.load(str(ae_ckpt), map_location=device, weights_only=False)
    team_dim = int(ck["team_dim"])
    latent_dim = int(ck["latent_dim"])
    hidden = tuple(int(x) for x in ck["hidden_dims"])
    ae = TeamAutoEncoder(team_dim, latent_dim, hidden).to(device)
    ae.load_state_dict(ck["state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae, team_dim, latent_dim, int(ck.get("rule_dim", 0))


@torch.inference_mode()
def encode_teams_with_ae(
    x: np.ndarray,
    ae_ckpt: str | Path,
    *,
    rule_vec: np.ndarray | None = None,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    dev = pick_device(str(device)) if not isinstance(device, torch.device) else device
    ae, team_dim, latent_dim, _ = load_frozen_encoder(ae_ckpt, dev)
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    teams = extract_team_matrix(x_np, team_dim)
    z = ae.encode(torch.from_numpy(teams).to(dev)).cpu().numpy()
    if rule_vec is not None and len(rule_vec) > 0:
        rv = np.asarray(rule_vec, dtype=np.float32)
        if rv.ndim == 1:
            rv = rv.reshape(1, -1)
        if rv.shape[0] == 1 and z.shape[0] > 1:
            rv = np.repeat(rv, z.shape[0], axis=0)
        if rv.shape[0] != z.shape[0]:
            raise ValueError(f"rule_vec rows {rv.shape[0]} != batch {z.shape[0]}")
        return np.concatenate([z, rv], axis=1).astype(np.float32)
    return z.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--autoencoder", type=str, required=True)
    ap.add_argument("--out", type=str, default="move_predictor_latent.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dims", type=str, default="256,128")
    ap.add_argument("--w_switch", type=float, default=1.0)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    d = np.load(args.data, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=np.float32)
    team_dim = int(np.asarray(d["team_encoder_dim"]).reshape(())) * 2
    rule_dim = int(d["rule_params"].shape[1]) if "rule_params" in d.files else max(0, X.shape[1] - team_dim)

    ae, _, latent_dim, _ = load_frozen_encoder(args.autoencoder, device)
    teams = extract_team_matrix(X, team_dim)
    with torch.inference_mode():
        Z = ae.encode(torch.from_numpy(teams).to(device)).cpu().numpy()
    if rule_dim > 0:
        Zin = np.concatenate([Z, np.asarray(d["rule_params"], dtype=np.float32)], axis=1)
    else:
        Zin = Z

    n = Zin.shape[0]
    perm = np.random.permutation(n)
    n_val = int(n * args.val_fraction)
    tr, va = perm[n_val:], perm[:n_val]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Zin[tr]), torch.from_numpy(Y[tr])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(Zin[va]), torch.from_numpy(Y[va])),
        batch_size=args.batch_size,
        shuffle=False,
    )

    hidden = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    model = MLPLocalHeads(Zin.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def forward_loss(xb: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, LossMetrics]:
        y0, y1, ysw = yb[:, 0:4], yb[:, 4:8], yb[:, 8:9]
        p0, p1, psw = model(xb)
        t0 = counts_to_move_dist(y0)
        t1 = counts_to_move_dist(y1)
        lp0 = -(t0 * F.log_softmax(p0, dim=-1)).sum(dim=-1).mean()
        lp1 = -(t1 * F.log_softmax(p1, dim=-1)).sum(dim=-1).mean()
        lsw = F.smooth_l1_loss(psw, ysw, beta=1.0)
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
        nb = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss_t, m = forward_loss(xb, yb)
            if train:
                opt.zero_grad()
                loss_t.backward()
                opt.step()
            for field in ("L_pkm0", "L_pkm1", "L_local", "L_switch", "L_total"):
                setattr(acc, field, getattr(acc, field) + getattr(m, field))
            nb += 1
        if nb:
            for field in ("L_pkm0", "L_pkm1", "L_local", "L_switch", "L_total"):
                setattr(acc, field, getattr(acc, field) / nb)
        return acc

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, True)
        with torch.inference_mode():
            va = run_epoch(val_loader, False)
        print(f"epoch {epoch}/{args.epochs}  train: {tr.format_line('local')}  val: {va.format_line('local')}")
        if va.L_total < best_val:
            best_val = va.L_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    out = Path(args.out)
    payload = {
        "model_type": "latent",
        "in_dim": int(Zin.shape[1]),
        "hidden_dims": hidden,
        "state_dict": model.state_dict(),
        "best_val_L_total": best_val,
        "latent_dim": latent_dim,
        "team_dim": team_dim,
        "rule_dim": rule_dim,
        "autoencoder_ckpt": str(Path(args.autoencoder).resolve()),
        "data_path": str(Path(args.data).resolve()),
        "move_count_dim": N_TARGETS,
    }
    torch.save(payload, out)
    out.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2),
        encoding="utf-8",
    )
    print(f"Guardado: {out.resolve()}  best_val_L_total={best_val:.5f}")


if __name__ == "__main__":
    main()
