"""Validação do previsor por simulação (reunião 2026-05-22).

Gera pares de equipas, corre batalhas com o agente configurado, obtém contagens
empíricas por slot (`run_battle_and_slot_counts`) e compara com a previsão da rede
com a mesma métrica KL(true || rede) que `eval_move_predictor_kl.py`.

Isto responde ao pedido do orientador: «correr a simulação para ver se o previsor
acertou» — o simulador substitui (ou complementa) o subset de validação do dataset.

Exemplo:
  python -u eval_move_predictor_sim.py \\
    --ckpt move_predictor_local.pt \\
    --n_team_pairs 32 \\
    --agent hybrid --agent_ckpt sup_guidelines_rules.pt \\
    --team_generator external --generator_path ../gen.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from eval_move_predictor_kl import (
    SLOT_LABELS,
    counts_to_prob,
    kl_divergence,
    predict_prob_9,
)
from move_predictor_io import build_x_from_teams, load_move_predictor, predict_move_counts
from vgc2.agent.battle import GreedyBattlePolicy, RandomBattlePolicy
from vgc2.competition.fixed_matches import FixedMatches
from vgc2.ml.battle_policies import make_self_play_pair
from vgc2.ml.external_generators import load_team_generator
from vgc2.ml.neural_policy import rule_feature_spec_from_ckpt
from vgc2.ml.rule_sampling import sample_battle_rule_params


def _params_to_vec(params, keys: tuple[str, ...]) -> np.ndarray:
    if not keys:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray([float(getattr(params, k)) for k in keys], dtype=np.float32)


def _unified_battle_params(agent_kind: str, ckpt_path: str):
    if agent_kind not in ("neural", "hybrid", "hybrid_neural_switch") or not str(ckpt_path).strip():
        from vgc2.battle_engine import BattleRuleParam

        return BattleRuleParam(), ()
    ck = torch.load(str(ckpt_path).strip(), map_location="cpu")
    keys, ranges = rule_feature_spec_from_ckpt(ck)
    if not keys:
        from vgc2.battle_engine import BattleRuleParam

        return BattleRuleParam(), ()
    return sample_battle_rule_params(keys, ranges), keys


def _make_agent(kind: str, ckpt: str, switch_path: str, battle_params, stochastic: bool):
    kind = kind.strip().lower()
    if kind == "greedy":
        p = GreedyBattlePolicy()
        p.set_params(battle_params)
        return p
    if kind == "random":
        p = RandomBattlePolicy()
        p.set_params(battle_params)
        return p
    if kind in ("neural", "hybrid", "hybrid_neural_switch"):
        mode = "hybrid_neural_switch" if kind in ("hybrid", "hybrid_neural_switch") else "neural"
        p0, _, _ = make_self_play_pair(
            mode,
            agent_ckpt=ckpt,
            switch_agent_path=switch_path,
            stochastic=stochastic,
        )
        p0.set_params(battle_params)
        return p0
    raise SystemExit(f"agent desconhecido: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser(description="KL(true||rede) em batalhas simuladas on-the-fly.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--n_team_pairs", type=int, default=32, help="Pares de equipas (×2 orientações).")
    ap.add_argument("--agent", choices=("greedy", "random", "neural", "hybrid"), default="hybrid")
    ap.add_argument("--agent_ckpt", type=str, default="sup_guidelines_rules.pt")
    ap.add_argument(
        "--switch_agent_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "battle_agent.py"),
    )
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--max_team_size", type=int, default=2)
    ap.add_argument("--max_pkm_moves", type=int, default=4)
    ap.add_argument("--team_generator", choices=("default", "external"), default="external")
    ap.add_argument("--generator_path", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--out_npz", type=str, default="", help="Opcional: guardar KL por batalha.")
    ap.add_argument("--also_kl_reverse", action="store_true")
    ap.add_argument("--skip_plots", action="store_true")
    ap.add_argument("--out_dir", type=str, default="plots_kl_sim")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    np.random.seed(int(args.seed))

    battle_params, rule_keys = _unified_battle_params(args.agent, args.agent_ckpt)
    rule_vec = _params_to_vec(battle_params, rule_keys)

    agent0 = _make_agent(args.agent, args.agent_ckpt, args.switch_agent_path, battle_params, bool(args.stochastic))
    agent1 = _make_agent("greedy", "", args.switch_agent_path, battle_params, False)

    team_gen = load_team_generator(args.team_generator, args.generator_path.strip() or None)
    fm = FixedMatches(
        (agent0, agent1),
        n_team_pairs=int(args.n_team_pairs),
        team_gen=team_gen,
        max_team_size=int(args.max_team_size),
        max_pkm_moves=int(args.max_pkm_moves),
    )
    fm.set_params(battle_params)
    fm.run(collect_counts=True)

    bundle = load_move_predictor(args.ckpt, args.device)

    emp_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    mae_list: list[float] = []

    for ti, (t0, t1) in enumerate(fm.team_pairs):
        for orientation in (0, 1):
            teams = (t0, t1) if orientation == 0 else (t1, t0)
            rollout_i = 2 * ti + orientation
            if rollout_i >= len(fm.counts_side0):
                continue
            emp = np.asarray(fm.counts_side0[rollout_i], dtype=np.float64)
            if emp.size != 9:
                continue
            x = build_x_from_teams(teams[0], teams[1], rule_vec=rule_vec, concat_rules=True)
            pred_counts = predict_move_counts(bundle, x)[0]

            emp_list.append(emp)
            pred_list.append(pred_counts)
            mae_list.append(float(np.abs(emp - pred_counts).mean()))

    if not emp_list:
        raise SystemExit("Sem batalhas com contagens 9-dim — verifica max_team_size=2 e max_pkm_moves=4.")

    emp_arr = np.stack(emp_list, axis=0)
    pred_arr = np.stack(pred_list, axis=0)
    p_true = counts_to_prob(emp_arr, eps=args.eps)
    q_pred = counts_to_prob(pred_arr, eps=args.eps)
    kl_main = np.maximum(kl_divergence(p_true, q_pred), 0.0)

    n_b = len(kl_main)
    print(f"simulação: n_battles={n_b}  agent={args.agent}  ckpt={Path(args.ckpt).name}")
    print(
        "KL(true || rede)  [empírico=sim, Q=rede]  "
        f"mean={kl_main.mean():.4f}  median={np.median(kl_main):.4f}  std={kl_main.std():.4f}  "
        f"min={kl_main.min():.4f}  max={kl_main.max():.4f}"
    )
    if mae_list:
        print(f"MAE contagens (bruto, não normalizado) mean={float(np.mean(mae_list)):.4f}")

    if args.also_kl_reverse:
        kl_rev = np.maximum(kl_divergence(q_pred, p_true), 0.0)
        print(f"KL(rede || true) mean={kl_rev.mean():.4f}")

    if args.out_npz:
        out_path = Path(args.out_npz)
        np.savez_compressed(
            out_path,
            kl_true_approx=kl_main.astype(np.float32),
            empirical_counts=emp_arr.astype(np.float32),
            predicted_prob=pred_arr.astype(np.float32),
            mae_per_battle=np.asarray(mae_list, dtype=np.float32),
            agent=str(args.agent),
            ckpt=str(Path(args.ckpt).resolve()),
            seed=int(args.seed),
        )
        print(f"guardado: {out_path.resolve()}")

    if args.skip_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("AVISO: matplotlib ausente — corre com pip install matplotlib para histograma KL.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(kl_main, bins=40, color="#4c72b0", alpha=0.85, edgecolor="white")
    ax.axvline(float(kl_main.mean()), color="#c44e52", linestyle="--", linewidth=2, label=f"média={kl_main.mean():.3f}")
    ax.axvline(float(np.median(kl_main)), color="#55a868", linestyle=":", linewidth=2, label=f"mediana={np.median(kl_main):.3f}")
    ax.set_xlabel("KL(true || rede)")
    ax.set_ylabel("nº batalhas")
    ax.set_title(f"Distribuição KL — simulação ({n_b} batalhas)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    hist_path = out_dir / "kl_histogram_sim.png"
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    print(f"plot: {hist_path.resolve()}")

    order = np.argsort(kl_main)
    from eval_move_predictor_kl import _plot_overlay, _plot_overlay_lines

    for label, idx in (("best", int(order[0])), ("worst", int(order[-1]))):
        k = float(kl_main[idx])
        title = f"SIM {label.upper()} — battle #{idx}  KL(true||rede)={k:.4f}"
        _plot_overlay(p_true[idx], q_pred[idx], out_dir / f"{label}_bars.png", title=title, kl_pq=k)
        _plot_overlay_lines(p_true[idx], q_pred[idx], out_dir / f"{label}_curves.png", title=title, kl_pq=k)
        print(f"  {label}: KL={k:.4f}  slots={SLOT_LABELS}")


if __name__ == "__main__":
    main()
