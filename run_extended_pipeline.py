"""Pipeline completo: vanilla + latent AE + sequência + demo balance."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="move_count_sup_switch.npz")
    ap.add_argument("--analytics_glob", type=str, default="moves_analytics_switch_v1_part*.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ae_epochs", type=int, default=40)
    ap.add_argument("--seq_epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--skip_vanilla", action="store_true")
    ap.add_argument("--skip_latent", action="store_true")
    ap.add_argument("--skip_sequence", action="store_true")
    ap.add_argument("--skip_balance", action="store_true")
    ap.add_argument("--seq_max_samples", type=int, default=80000)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    data = root / args.data
    if not data.is_file():
        raise SystemExit(f"Falta {data}")

    common = ["--data", args.data, "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--device", args.device]

    if not args.skip_vanilla:
        _run([py, "-u", "run_move_predictor_pipeline.py", *common], root)

    ae_path = root / "team_autoencoder.pt"
    latent_path = root / "move_predictor_latent.pt"
    if not args.skip_latent:
        _run(
            [
                py,
                "-u",
                "train_team_autoencoder.py",
                "--data",
                args.data,
                "--out",
                str(ae_path),
                "--epochs",
                str(args.ae_epochs),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
            ],
            root,
        )
        _run(
            [
                py,
                "-u",
                "train_move_count_predictor_latent.py",
                "--data",
                args.data,
                "--autoencoder",
                str(ae_path),
                "--out",
                str(latent_path),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
            ],
            root,
        )
        _run(
            [py, "-u", "eval_move_count_predictor.py", "--ckpt", str(latent_path), "--data", args.data, "--device", args.device],
            root,
        )

    seq_data = root / "rollout_seq_sup.npz"
    if not args.skip_sequence:
        _run(
            [
                py,
                "-u",
                "build_rollout_sequence_dataset.py",
                "--glob",
                args.analytics_glob,
                "--out",
                str(seq_data),
                "--concat_rules_to_x",
                "--max_samples",
                str(args.seq_max_samples),
            ],
            root,
        )
        _run(
            [
                py,
                "-u",
                "train_rollout_sequence_predictor.py",
                "--data",
                str(seq_data),
                "--epochs",
                str(args.seq_epochs),
                "--batch_size",
                str(min(256, args.batch_size)),
                "--device",
                args.device,
            ],
            root,
        )

    if not args.skip_balance:
        _run(
            [
                py,
                "-u",
                "run_balance_rollouts.py",
                "--agent0",
                "hybrid",
                "--agent1",
                "greedy",
                "--move_predictor_ckpt",
                str(root / "move_predictor_local.pt"),
                "--team_generator",
                "external",
                "--generator_path",
                str(root.parent / "gen.py"),
            ],
            root,
        )

    if not args.skip_vanilla:
        _run(
            [
                py,
                "-u",
                "eval_move_predictor_kl.py",
                "--ckpt",
                str(root / "move_predictor_local.pt"),
                "--data",
                args.data,
                "--device",
                args.device,
                "--out_dir",
                str(root / "plots_kl_local"),
            ],
            root,
        )

    print("\nOK: extended pipeline concluido.")


if __name__ == "__main__":
    main()
