"""Pipeline completo alinhado com reunião 2026-05-15.

1) Estatísticas do dataset (opcional)
2) Treino L_local (recomendado) + avaliação
3) Treino L_global (baseline) + avaliação

Pré-requisito: `move_count_sup_big.npz` (build_move_count_supervision.py).

Uso:
  python -u run_move_predictor_pipeline.py --data move_count_sup_big.npz
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline previsor de moves (reunião 2026-05-15).")
    ap.add_argument("--data", type=str, default="move_count_sup_big.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--skip_stats", action="store_true")
    ap.add_argument("--skip_global", action="store_true", help="Só treina L_local (modo recomendado).")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data = root / args.data
    if not data.is_file():
        raise SystemExit(
            f"Falta {data}. Gera com:\n"
            "  build_move_count_supervision.py --glob moves_analytics_big_v2_part*.npz "
            "--out move_count_sup_big.npz --concat_rules_to_x"
        )

    py = sys.executable
    common = ["--data", str(data), "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--device", args.device]

    if not args.skip_stats:
        _run([py, "-u", "stats_move_usage.py", "--npz", str(data)], root)

    out_local = root / "move_predictor_local.pt"
    _run(
        [py, "-u", "train_move_count_predictor.py", *common, "--loss", "local", "--switch_loss", "mse", "--w_switch", "1.0", "--out", str(out_local)],
        root,
    )
    _run([py, "-u", "eval_move_count_predictor.py", "--ckpt", str(out_local), "--data", str(data), "--device", args.device], root)
    _run(
        [
            py,
            "-u",
            "eval_move_predictor_kl.py",
            "--ckpt",
            str(out_local),
            "--data",
            str(data),
            "--device",
            args.device,
            "--out_dir",
            str(root / "plots_kl_local"),
        ],
        root,
    )

    if not args.skip_global:
        out_global = root / "move_predictor_global.pt"
        _run(
            [py, "-u", "train_move_count_predictor.py", *common, "--loss", "global", "--out", str(out_global)],
            root,
        )
        _run([py, "-u", "eval_move_count_predictor.py", "--ckpt", str(out_global), "--data", str(data), "--device", args.device], root)

    print("\nOK: pipeline concluido.")
    print(f"  modelo principal (L_local): {out_local}")
    if not args.skip_global:
        print(f"  baseline (L_global):       {root / 'move_predictor_global.pt'}")


if __name__ == "__main__":
    main()
