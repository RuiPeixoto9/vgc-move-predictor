"""Dataset para prever sequência de moves (lado 0) a partir das equipas iniciais."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

PAD_ID = -999
SWITCH_CLASS = 4


def _id_to_class(mid: int) -> int:
    if mid == PAD_ID:
        return -100  # ignore em CE
    if mid == -1:
        return SWITCH_CLASS
    return int(mid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, required=True)
    ap.add_argument("--out", type=str, default="rollout_seq_sup.npz")
    ap.add_argument("--max_len", type=int, default=24)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = todos")
    ap.add_argument("--concat_rules_to_x", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"Sem ficheiros para {args.glob!r}")

    xs, seqs, lens, rules = [], [], [], []
    t_max = int(args.max_len)

    for p in paths:
        d = np.load(p, allow_pickle=True)
        X = d["X"]
        ids = d["rollout_move_ids"]
        rlen = d["rollout_len"]
        rule_params = d["rule_params"] if "rule_params" in d.files else None
        for i in range(X.shape[0]):
            if args.max_samples and len(xs) >= args.max_samples:
                break
            t_len = min(int(rlen[i]), t_max)
            row = np.full((t_max,), -100, dtype=np.int64)
            for t in range(t_len):
                row[t] = _id_to_class(int(ids[i, t, 0]))
            xi = np.asarray(X[i], dtype=np.float32)
            if args.concat_rules_to_x and rule_params is not None:
                xi = np.concatenate([xi, np.asarray(rule_params[i], dtype=np.float32)], axis=0)
            xs.append(xi)
            seqs.append(row)
            lens.append(t_len)
            rules.append(np.asarray(rule_params[i], dtype=np.float32) if rule_params is not None else np.zeros((0,)))
        if args.max_samples and len(xs) >= args.max_samples:
            break

    X_out = np.stack(xs, axis=0)
    S_out = np.stack(seqs, axis=0)
    L_out = np.asarray(lens, dtype=np.int64)
    R_out = np.stack(rules, axis=0) if rules and rules[0].size else np.zeros((len(xs), 0), dtype=np.float32)

    out = Path(args.out)
    np.savez_compressed(
        out,
        X=X_out,
        Y_seq=S_out,
        seq_len=L_out,
        rule_params=R_out,
        max_len=t_max,
        n_classes=5,
        pad_class=-100,
        switch_class=SWITCH_CLASS,
        kind="rollout_sequence_supervision",
    )
    print(f"Guardado {out.resolve()}  X={X_out.shape}  Y_seq={S_out.shape}  samples={X_out.shape[0]}")


if __name__ == "__main__":
    main()
