"""Pipeline guidelines: regras generalizadas (4 knobs) + gen.py + IL + move predictor + DQN + SAC + eval.

Usa `vgc2.ml.guidelines_rules.GUIDELINE_RULE_FEATURES_CLI` (STAB, WEATHER_BOOST, screens).

Subcomandos:
  datasets   — gera `ds_guidelines_rules*.npz` e `moves_guidelines_rules*.npz`
  train-il   — `train_supervised_actor` no dataset de transição
  train-move — previsor de moves (supervisionado, --w_value 0)
  train-rl   — DQN + SAC com mesmas `--rule_features`
  eval       — `sweep_checkpoints` nos artefactos deste pipeline
  all        — corre sequência (use `--quick` para smoke rápido)

Exemplos:
    python -u run_guidelines_generalized.py all --quick --generator_path \".../gen.py\"
    python -u run_guidelines_generalized.py all --long-rl --generator_path \".../gen.py\"
    python -u run_guidelines_generalized.py train-rl --long-rl ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from vgc2.ml.guidelines_rules import GUIDELINE_RULE_FEATURES_CLI

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _default_generator_path() -> Path:
    cand = ROOT.parent / "gen.py"
    return cand if cand.is_file() else Path("gen.py")


def _run(args: list[str]) -> None:
    cmd = [PY, "-u", *args]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def cmd_datasets(ns: argparse.Namespace) -> None:
    gen = Path(ns.generator_path).resolve()
    rule = ns.rule_features or GUIDELINE_RULE_FEATURES_CLI
    _run(
        [
            str(ROOT / "gen_transition_dataset.py"),
            "--episodes",
            str(int(ns.episodes_transition)),
            "--out",
            str(ROOT / ns.out_transition),
            "--seed",
            str(int(ns.seed)),
            "--encoder",
            "greedy_subset",
            "--n_active",
            "1",
            "--max_team_size",
            "2",
            "--max_pkm_moves",
            "4",
            "--opponent",
            "greedy",
            "--expert",
            "greedy",
            "--team_generator",
            "external",
            "--generator_path",
            str(gen),
            "--rule_features",
            rule,
            "--resample_rules_each_episode",
        ]
    )
    _run(
        [
            str(ROOT / "gen_move_dataset.py"),
            "--episodes",
            str(int(ns.episodes_moves)),
            "--out",
            str(ROOT / ns.out_moves),
            "--seed",
            str(int(ns.seed) + 1),
            "--encoder",
            "greedy_subset",
            "--n_active",
            "1",
            "--max_team_size",
            "2",
            "--max_pkm_moves",
            "4",
            "--agent_policy",
            "greedy",
            "--opp_label",
            "greedy",
            "--team_generator",
            "external",
            "--generator_path",
            str(gen),
            "--rule_features",
            rule,
            "--resample_rules_each_episode",
        ]
    )


def cmd_train_il(ns: argparse.Namespace) -> None:
    gen = Path(ns.generator_path).resolve()
    rule = ns.rule_features or GUIDELINE_RULE_FEATURES_CLI
    init = (ROOT / ns.init_il).resolve() if ns.init_il else None
    if not init or not init.is_file():
        raise SystemExit(f"init IL inexistente: {init}")
    _run(
        [
            str(ROOT / "train_supervised_actor.py"),
            "--dataset",
            str(ROOT / ns.out_transition),
            "--init_ckpt",
            str(init),
            "--encoder",
            "greedy_subset",
            "--n_active",
            "1",
            "--max_team_size",
            "2",
            "--max_pkm_moves",
            "4",
            "--rule_features",
            rule,
            "--resample_rules_each_episode",
            "--hidden_dims",
            ns.hidden_dims,
            "--w_value",
            "0",
            "--epochs",
            str(int(ns.epochs_il)),
            "--lr",
            str(ns.lr_il),
            "--batch_size",
            str(int(ns.batch_il)),
            "--weight_decay",
            "1e-5",
            "--out",
            str(ROOT / ns.out_il),
            "--device",
            ns.device,
        ]
    )


def cmd_train_move(ns: argparse.Namespace) -> None:
    rule = ns.rule_features or GUIDELINE_RULE_FEATURES_CLI
    il_ckpt = (ROOT / ns.out_il).resolve()
    if not il_ckpt.is_file():
        raise SystemExit(f"Treina IL primeiro ou indica --out-il existente: {il_ckpt}")
    _run(
        [
            str(ROOT / "train_supervised_actor.py"),
            "--dataset",
            str(ROOT / ns.out_moves),
            "--init_ckpt",
            str(il_ckpt),
            "--encoder",
            "greedy_subset",
            "--n_active",
            "1",
            "--max_team_size",
            "2",
            "--max_pkm_moves",
            "4",
            "--rule_features",
            rule,
            "--resample_rules_each_episode",
            "--hidden_dims",
            ns.hidden_dims,
            "--w_value",
            "0",
            "--epochs",
            str(int(ns.epochs_move)),
            "--lr",
            str(ns.lr_move),
            "--batch_size",
            str(int(ns.batch_move)),
            "--weight_decay",
            "1e-5",
            "--out",
            str(ROOT / ns.out_move_pred),
            "--device",
            ns.device,
        ]
    )


def cmd_train_rl(ns: argparse.Namespace) -> None:
    gen = Path(ns.generator_path).resolve()
    rule = ns.rule_features or GUIDELINE_RULE_FEATURES_CLI
    il_ckpt = (ROOT / ns.out_il).resolve()
    if not il_ckpt.is_file():
        raise SystemExit(f"Checkpoint IL com regras inexistente: {il_ckpt}")
    _run(
        [
            str(ROOT / "train_dqn_battle.py"),
            "--total_timesteps",
            str(int(ns.timesteps_dqn)),
            "--init_ckpt",
            str(il_ckpt),
            "--encoder",
            "greedy_subset",
            "--n_active",
            "1",
            "--max_team_size",
            "2",
            "--max_pkm_moves",
            "4",
            "--opponent",
            "greedy",
            "--team_generator",
            "external",
            "--generator_path",
            str(gen),
            "--rule_features",
            rule,
            "--resample_rules_each_episode",
            "--dense_reward",
            "--hidden_dims",
            ns.hidden_dims,
            "--learning_rate",
            str(ns.lr_dqn),
            "--out",
            str(ROOT / ns.out_dqn),
            "--device",
            ns.device,
        ]
    )
    _run(
        [
            str(ROOT / "train_sac_battle.py"),
            "--total_timesteps",
            str(int(ns.timesteps_sac)),
            "--init_ckpt",
            str(il_ckpt),
            "--encoder",
            "greedy_subset",
            "--n_active",
            "1",
            "--max_team_size",
            "2",
            "--max_pkm_moves",
            "4",
            "--opponent",
            "greedy",
            "--team_generator",
            "external",
            "--generator_path",
            str(gen),
            "--rule_features",
            rule,
            "--resample_rules_each_episode",
            "--dense_reward",
            "--hidden_dims",
            ns.hidden_dims,
            "--out",
            str(ROOT / ns.out_sac),
            "--device",
            ns.device,
        ]
    )


def cmd_eval(ns: argparse.Namespace) -> None:
    gen = Path(ns.generator_path).resolve()
    globs = ";".join(ns.eval_globs)
    _run(
        [
            str(ROOT / "sweep_checkpoints.py"),
            "--glob",
            globs,
            "--games",
            str(int(ns.eval_games)),
            "--seeds",
            ns.eval_seeds,
            "--opponent",
            "greedy",
            "--team_generator",
            "external",
            "--generator_path",
            str(gen),
            "--top_k",
            "20",
        ]
    )


def cmd_all(ns: argparse.Namespace) -> None:
    cmd_datasets(ns)
    cmd_train_il(ns)
    cmd_train_move(ns)
    cmd_train_rl(ns)
    cmd_eval(ns)


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generator_path",
        type=str,
        default="",
        help="gen.py do orientador (default: ../gen.py relativo ao pokemon-vgc-engine).",
    )
    parser.add_argument(
        "--rule_features",
        type=str,
        default="",
        help="Override; default = GUIDELINE_RULE_FEATURES_CLI.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden_dims", type=str, default="128,128,64")
    parser.add_argument("--quick", action="store_true", help="Smoke: poucos episódios / passos.")
    parser.add_argument(
        "--long-rl",
        action="store_true",
        help="Sem --quick: DQN/SAC com mais passos (300k cada).",
    )

    parser.add_argument("--out-transition", type=str, default="ds_guidelines_rules.npz")
    parser.add_argument("--out-moves", type=str, default="moves_guidelines_rules.npz")
    parser.add_argument("--out-il", type=str, default="sup_guidelines_rules.pt")
    parser.add_argument("--out-move-pred", type=str, default="move_pred_guidelines_rules.pt")
    parser.add_argument("--out-dqn", type=str, default="dqn_guidelines_rules.pt")
    parser.add_argument("--out-sac", type=str, default="sac_guidelines_rules.pt")
    parser.add_argument("--init-il", type=str, default="sup_greedy_subset_ep100.pt")

    parser.add_argument("--episodes-transition", type=int, default=1800)
    parser.add_argument("--episodes-moves", type=int, default=1500)
    parser.add_argument("--epochs-il", type=int, default=18)
    parser.add_argument("--epochs-move", type=int, default=15)
    parser.add_argument("--lr-il", type=float, default=5e-5)
    parser.add_argument("--lr-move", type=float, default=5e-5)
    parser.add_argument("--batch-il", type=int, default=1024)
    parser.add_argument("--batch-move", type=int, default=1024)
    parser.add_argument("--timesteps-dqn", type=int, default=120000)
    parser.add_argument("--timesteps-sac", type=int, default=120000)
    parser.add_argument("--lr-dqn", type=float, default=3e-5)

    parser.add_argument("--eval-globs", nargs="+", default=[])
    parser.add_argument("--eval-games", type=int, default=400)
    parser.add_argument("--eval-seeds", type=str, default="0,1,2")


def main() -> None:
    parent = argparse.ArgumentParser(add_help=False)
    _common_args(parent)

    p = argparse.ArgumentParser(description="Pipeline guidelines: regras generalizadas.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_d = sub.add_parser("datasets", parents=[parent], help="Gera datasets de transição + moves.")
    sp_d.set_defaults(func=cmd_datasets)

    sp_i = sub.add_parser("train-il", parents=[parent], help="Treina IL com regras.")
    sp_i.set_defaults(func=cmd_train_il)

    sp_m = sub.add_parser("train-move", parents=[parent], help="Treina previsor de moves.")
    sp_m.set_defaults(func=cmd_train_move)

    sp_r = sub.add_parser("train-rl", parents=[parent], help="Treina DQN + SAC com regras.")
    sp_r.set_defaults(func=cmd_train_rl)

    sp_e = sub.add_parser("eval", parents=[parent], help="Sweep winrate nos checkpoints indicados.")
    sp_e.set_defaults(func=cmd_eval)

    sp_a = sub.add_parser(
        "all",
        parents=[parent],
        help="datasets + train-il + train-move + train-rl + eval",
    )
    sp_a.set_defaults(func=cmd_all)

    args = p.parse_args()
    if not str(args.generator_path).strip():
        args.generator_path = str(_default_generator_path())
    if args.quick:
        args.out_transition = "ds_guidelines_rules_quick.npz"
        args.out_moves = "moves_guidelines_rules_quick.npz"
        args.out_il = "sup_guidelines_rules_quick.pt"
        args.out_move_pred = "move_pred_guidelines_rules_quick.pt"
        args.out_dqn = "dqn_guidelines_rules_quick.pt"
        args.out_sac = "sac_guidelines_rules_quick.pt"
        args.episodes_transition = 120
        args.episodes_moves = 100
        args.epochs_il = 4
        args.epochs_move = 4
        args.timesteps_dqn = 8000
        args.timesteps_sac = 8000
        args.eval_games = 200
    elif getattr(args, "long_rl", False):
        args.timesteps_dqn = 300000
        args.timesteps_sac = 300000
    if not args.eval_globs:
        args.eval_globs = [
            str(ROOT / args.out_il),
            str(ROOT / args.out_move_pred),
            str(ROOT / args.out_dqn),
            str(ROOT / args.out_sac),
        ]
    args.func(args)
    print("OK:", args.cmd)


if __name__ == "__main__":
    main()
