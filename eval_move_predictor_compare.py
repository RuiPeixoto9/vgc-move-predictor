"""Compara vários checkpoints no mesmo split de validação (KL + MAE).

Útil para experimentar outro previsor e repetir a validação (reunião 2026-05-22).

Exemplo:
  python -u eval_move_predictor_compare.py \\
    --data move_count_sup_switch.npz \\
    --ckpts move_predictor_local.pt move_predictor_global.pt move_predictor_latent.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval_move_predictor_kl import _val_split, counts_to_prob, kl_divergence, predict_prob_9
from move_predictor_io import load_move_predictor, predict_move_counts


def _eval_one(ckpt: str, X_val: np.ndarray, Y_val: np.ndarray, *, device: str, eps: float) -> dict[str, float]:
    bundle = load_move_predictor(ckpt, device)
    pred_counts = predict_move_counts(bundle, X_val)
    mae = float(np.abs(pred_counts - Y_val).mean())
    p_true = counts_to_prob(Y_val, eps=eps)
    q_pred = predict_prob_9(bundle, X_val, eps=eps)
    # Mesmo critério que eval_move_predictor_kl.py (eps Laplace também no clip do KL)
    kl = np.maximum(kl_divergence(p_true, q_pred, eps=eps), 0.0)
    kl_rev = np.maximum(kl_divergence(q_pred, p_true, eps=eps), 0.0)
    return {
        "model": bundle["model_type"],
        "mae": mae,
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_std": float(kl.std()),
        "kl_rev_mean": float(kl_rev.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Tabela KL/MAE para vários .pt no mesmo val set.")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpts", type=str, nargs="+", required=True)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--out_csv", type=str, default="", help="Opcional: guardar tabela CSV.")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=np.float32)
    val_idx = _val_split(X.shape[0], args.val_fraction, args.seed)
    Xv, Yv = X[val_idx], Y[val_idx]
    print(f"data={Path(args.data).name}  val_n={len(val_idx)}  seed={args.seed}")
    print(f"{'ckpt':<32} {'type':<8} {'MAE':>8} {'KL mean':>10} {'KL med':>10} {'KL std':>8} {'KL rev':>10}")
    print("-" * 92)

    rows: list[dict[str, str | float]] = []
    for ckpt in args.ckpts:
        path = Path(ckpt)
        if not path.is_file():
            print(f"{path.name:<32}  MISSING")
            continue
        try:
            m = _eval_one(str(path), Xv, Yv, device=args.device, eps=args.eps)
        except Exception as exc:
            print(f"{path.name:<32}  ERRO: {exc}")
            continue
        print(
            f"{path.name:<32} {m['model']:<8} {m['mae']:8.4f} {m['kl_mean']:10.4f} "
            f"{m['kl_median']:10.4f} {m['kl_std']:8.4f} {m['kl_rev_mean']:10.4f}"
        )
        rows.append({"ckpt": path.name, **m})

    if args.out_csv and rows:
        import csv

        out = Path(args.out_csv)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"csv: {out.resolve()}")


if __name__ == "__main__":
    main()
