"""Converte shards `moves_analytics` (com `move_counts_side0/1`) num dataset de treino com augmentation.

Augmentação (reuniões 2026-05-08 / 2026-05-15, notas orientador):
  - **X,Y -> Xm**: T0||T1, target = contagens de moves de T0
  - **Y,X -> Ym**: T1||T0 (swap), target = contagens de moves de T1
  - Alvo Y: [2 pkm x 4 moves] + [1 switch global] = 9 valores

Cada batalha origina **2 linhas** de treino.

Requisitos nos `.npz` de entrada:
  - `X`, `rule_params` (opcional), `move_counts_side0`, `move_counts_side1`, `team_encoder_dim`

Exemplo:
  python -u build_move_count_supervision.py \\
    --glob "moves_analytics_big_part*.npz" \\
    --out move_count_sup_big.npz \\
    --concat_rules_to_x
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def _swap_team_halves(x_row: np.ndarray, team_dim: int) -> np.ndarray:
    if x_row.ndim != 1:
        raise ValueError("Esperado vetor 1D")
    if x_row.shape[0] != 2 * team_dim:
        raise ValueError(f"X.shape[1]={x_row.shape[0]} != 2*team_dim={2 * team_dim}")
    out = np.empty_like(x_row)
    out[:team_dim] = x_row[team_dim:]
    out[team_dim:] = x_row[:team_dim]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset supervisionado: contagens de moves + augmentation T0/T1.")
    ap.add_argument("--glob", type=str, default="", help='Glob de shards, ex. "moves_analytics_big_part*.npz"')
    ap.add_argument(
        "--inputs",
        type=str,
        default="",
        help="Alias de --glob (compatibilidade).",
    )
    ap.add_argument("--out", type=str, required=True, help="Ficheiro .npz de saída (tudo em memória).")
    ap.add_argument(
        "--concat_rules_to_x",
        action="store_true",
        help="Se definido, X_out = [X_teams, rule_params] por linha.",
    )
    args = ap.parse_args()

    glob_pat = str(args.glob or args.inputs).strip()
    if not glob_pat:
        raise SystemExit("Indica --glob ou --inputs.")
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"Nenhum ficheiro para glob: {glob_pat!r}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rules: list[np.ndarray] = []
    aug_tag: list[np.ndarray] = []
    winners: list[np.ndarray] = []

    team_dim_ref: int | None = None
    move_dim_ref: int | None = None
    rule_cols_ref: int | None = None

    for p in paths:
        data = np.load(p, allow_pickle=True)
        if "move_counts_side0" not in data.files or "move_counts_side1" not in data.files:
            raise SystemExit(
                f"{p} não tem move_counts_side0/side1. Volta a gerar com `gen_moves_analytics_dataset.py` atualizado."
            )
        X = data["X"]
        c0 = data["move_counts_side0"]
        c1 = data["move_counts_side1"]
        team_dim = int(np.asarray(data["team_encoder_dim"]).reshape(()))
        rule_params = data["rule_params"] if "rule_params" in data.files else None

        if team_dim_ref is None:
            team_dim_ref = team_dim
        elif team_dim_ref != team_dim:
            raise SystemExit(f"team_encoder_dim inconsistente: {team_dim_ref} vs {team_dim} em {p}")

        if move_dim_ref is None:
            move_dim_ref = int(c0.shape[1])
        elif move_dim_ref != int(c0.shape[1]):
            raise SystemExit(f"move_count dim inconsistente: {move_dim_ref} vs {c0.shape[1]} em {p}")

        if rule_params is not None:
            if rule_cols_ref is None:
                rule_cols_ref = int(rule_params.shape[1])
            elif rule_cols_ref != int(rule_params.shape[1]):
                raise SystemExit(f"rule_params cols inconsistentes em {p}")

        w = data["winner"] if "winner" in data.files else np.full(X.shape[0], -1, dtype=np.int64)

        for i in range(X.shape[0]):
            xi = np.asarray(X[i], dtype=np.float32)
            ri = np.asarray(rule_params[i], dtype=np.float32) if rule_params is not None else np.zeros((0,), dtype=np.float32)

            # Orientação A: T0,T1 -> target contagens T0
            xs.append(xi)
            ys.append(np.asarray(c0[i], dtype=np.float32))
            rules.append(ri)
            aug_tag.append(np.int64(0))
            winners.append(np.int64(w[i]))

            # Orientação B: T1,T0 -> target contagens T1
            xs.append(_swap_team_halves(xi, team_dim))
            ys.append(np.asarray(c1[i], dtype=np.float32))
            rules.append(ri)
            aug_tag.append(np.int64(1))
            winners.append(np.int64(w[i]))

    X_stacked = np.stack(xs, axis=0)
    Y_stacked = np.stack(ys, axis=0)
    aug_stacked = np.stack(aug_tag, axis=0)
    win_stacked = np.stack(winners, axis=0)
    R_stacked = np.stack(rules, axis=0) if rule_cols_ref not in (None, 0) else np.zeros((X_stacked.shape[0], 0), dtype=np.float32)

    if args.concat_rules_to_x:
        if R_stacked.shape[1] == 0:
            raise SystemExit("--concat_rules_to_x mas não há rule_params nos shards.")
        X_out = np.concatenate([X_stacked, R_stacked], axis=1).astype(np.float32)
    else:
        X_out = X_stacked

    out_path = Path(args.out)
    np.savez_compressed(
        out_path,
        X=X_out,
        Y=Y_stacked,
        rule_params=R_stacked,
        augment_orientation=aug_stacked,
        winner=win_stacked,
        team_encoder_dim=int(team_dim_ref or 0),
        move_count_dim=int(move_dim_ref or 0),
        concat_rules_to_x=bool(args.concat_rules_to_x),
        kind="move_count_supervision",
        sources=np.array(";".join(paths), dtype=object),
    )
    print(
        f"Guardado {out_path.resolve()}\n"
        f"  X_out={X_out.shape}  Y={Y_stacked.shape}  samples={X_out.shape[0]} (2x batalhas)\n"
        f"  team_encoder_dim={team_dim_ref}  move_count_dim={move_dim_ref}"
    )


if __name__ == "__main__":
    main()
