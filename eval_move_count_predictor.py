"""Avalia checkpoint de `train_move_count_predictor.py` no conjunto de validação."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from move_predictor_config import N_TARGETS, PKM0_SLICE, PKM1_SLICE, SWITCH_SLICE
from move_predictor_io import load_move_predictor, predict_move_counts
from train_move_count_predictor import (
    MLPGlobal,
    MLPLocalHeads,
    counts_to_move_dist,
    pick_device,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    d = np.load(args.data, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=np.float32)

    n = X.shape[0]
    perm = np.random.permutation(n)
    n_val = int(n * args.val_fraction)
    idx = perm[:n_val]
    kind = str(ck["model_type"])

    if kind == "latent":
        bundle = load_move_predictor(args.ckpt, device)
        pred = predict_move_counts(bundle, X[idx])
        yv = Y[idx]
        mae_all = np.abs(pred - yv).mean(axis=0)
        print(f"model=latent  val_n={len(idx)}  MAE_mean={float(mae_all.mean()):.4f}")
        print(f"  MAE pkm0 slots: {mae_all[PKM0_SLICE]}")
        print(f"  MAE pkm1 slots: {mae_all[PKM1_SLICE]}")
        print(f"  MAE switch:     {mae_all[SWITCH_SLICE].item():.4f}")
        return

    Xv = torch.from_numpy(X[idx]).to(device)
    Yv = torch.from_numpy(Y[idx]).to(device)

    hidden = tuple(int(x) for x in ck["hidden_dims"])
    in_dim = int(ck["in_dim"])

    if kind == "global":
        model = MLPGlobal(in_dim, hidden, N_TARGETS).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            pred = model(Xv)
        mae_all = (pred - Yv).abs().mean(dim=0).cpu().numpy()
        mae_mean = float(mae_all.mean())
        print(f"model=global  val_n={len(idx)}  MAE_mean={mae_mean:.4f}")
        print(f"  MAE pkm0 slots: {mae_all[PKM0_SLICE]}")
        print(f"  MAE pkm1 slots: {mae_all[PKM1_SLICE]}")
        print(f"  MAE switch:     {mae_all[SWITCH_SLICE].item():.4f}")
        return

    model = MLPLocalHeads(in_dim, hidden).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        p0, p1, psw = model(Xv)
        dist0 = F.softmax(p0, dim=-1)
        dist1 = F.softmax(p1, dim=-1)
        # Contagem esperada sob a distribuição prevista (para comparar com counts)
        tot0 = Yv[:, PKM0_SLICE].sum(dim=-1, keepdim=True).clamp(min=1e-6)
        tot1 = Yv[:, PKM1_SLICE].sum(dim=-1, keepdim=True).clamp(min=1e-6)
        exp0 = dist0 * tot0
        exp1 = dist1 * tot1
        mae_p0 = (exp0 - Yv[:, PKM0_SLICE]).abs().mean().item()
        mae_p1 = (exp1 - Yv[:, PKM1_SLICE]).abs().mean().item()
        mae_sw = (psw - Yv[:, SWITCH_SLICE]).abs().mean().item()
        # CE alvo suave (mesma métrica de treino)
        t0 = counts_to_move_dist(Yv[:, PKM0_SLICE])
        t1 = counts_to_move_dist(Yv[:, PKM1_SLICE])
        ce0 = -(t0 * F.log_softmax(p0, dim=-1)).sum(dim=-1).mean().item()
        ce1 = -(t1 * F.log_softmax(p1, dim=-1)).sum(dim=-1).mean().item()
    print(f"model=local  val_n={len(idx)}")
    print(f"  MAE expected-count pkm0={mae_p0:.4f}  pkm1={mae_p1:.4f}  switch={mae_sw:.4f}")
    print(f"  CE_soft pkm0={ce0:.4f}  pkm1={ce1:.4f}")


if __name__ == "__main__":
    main()
