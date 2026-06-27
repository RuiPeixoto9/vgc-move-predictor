"""Rollouts `FixedMatches` + métricas `evaluate_rules` e previsor de moves.

Integração (reunião 2026-05-15):
  - `evaluate_rules`: distribuição switch/damage/effect do vencedor (alvo 20/60/20)
  - Com `--move_predictor_ckpt`: compara previsões vs contagens empíricas e perfil 20/60/20 rápido

Exemplo:
  python -u run_balance_rollouts.py \\
    --agent0 hybrid --agent0_ckpt sup_guidelines_rules.pt \\
    --agent1 greedy --move_predictor_ckpt move_predictor_local.pt \\
    --n_team_pairs 8 --team_generator external --generator_path ../gen.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from eval_move_predictor_kl import counts_to_prob, kl_divergence, predict_prob_9
from move_predictor_io import (
    action_profile_from_counts,
    balance_score_from_profile,
    build_x_from_teams,
    load_move_predictor,
    predict_move_counts,
)
from vgc2.agent import BattlePolicy
from vgc2.agent.battle import GreedyBattlePolicy, RandomBattlePolicy
from vgc2.balance.rules.evaluator import evaluate_rules
from vgc2.balance.rules.predictor_evaluator import evaluate_combined_balance
from vgc2.battle_engine import BattleRuleParam
from vgc2.competition.fixed_matches import FixedMatches
from vgc2.ml.battle_policies import make_self_play_pair
from vgc2.ml.external_generators import load_team_generator
from vgc2.ml.neural_policy import inference_battle_policy_from_ckpt, rule_feature_spec_from_ckpt
from vgc2.ml.rule_sampling import sample_battle_rule_params


def _params_to_vec(params: BattleRuleParam, keys: tuple[str, ...]) -> np.ndarray:
    if not keys:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray([float(getattr(params, k)) for k in keys], dtype=np.float32)


def _unified_battle_params(pairs: list[tuple[str, str, str]]) -> tuple[BattleRuleParam, tuple[str, ...]]:
    spec: tuple[tuple[str, ...], dict[str, tuple[float, float]]] | None = None
    for kind, ckpt_path, _ in pairs:
        if kind not in ("neural", "hybrid", "hybrid_neural_switch") or not str(ckpt_path).strip():
            continue
        ck = torch.load(str(ckpt_path).strip(), map_location="cpu")
        keys, ranges = rule_feature_spec_from_ckpt(ck)
        if not keys:
            continue
        if spec is None:
            spec = (keys, ranges)
        elif spec[0] != keys:
            raise ValueError("rule_feature_keys diferentes entre checkpoints.")
    if spec is None:
        return BattleRuleParam(), ()
    return sample_battle_rule_params(spec[0], spec[1]), spec[0]


def _make_side(
    kind: str,
    ckpt_path: str,
    switch_path: str,
    battle_params: BattleRuleParam,
    stochastic: bool,
) -> BattlePolicy:
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
            agent_ckpt=ckpt_path,
            switch_agent_path=switch_path,
            stochastic=stochastic,
        )
        p0.set_params(battle_params)
        return p0
    raise SystemExit(f"agent desconhecido: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser(description="evaluate_rules + integração move predictor.")
    ap.add_argument("--agent0", choices=("greedy", "random", "neural", "hybrid"), default="hybrid")
    ap.add_argument("--agent1", choices=("greedy", "random", "neural", "hybrid"), default="greedy")
    ap.add_argument("--agent0_ckpt", type=str, default="sup_guidelines_rules.pt")
    ap.add_argument("--agent1_ckpt", type=str, default="")
    ap.add_argument(
        "--switch_agent_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "battle_agent.py"),
    )
    ap.add_argument("--move_predictor_ckpt", type=str, default="", help="move_predictor_local.pt ou latent.")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--n_team_pairs", type=int, default=8)
    ap.add_argument("--max_team_size", type=int, default=2)
    ap.add_argument("--max_pkm_moves", type=int, default=4)
    ap.add_argument("--team_generator", choices=("default", "external"), default="external")
    ap.add_argument("--generator_path", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    np.random.seed(int(args.seed))

    battle_params, rule_keys = _unified_battle_params(
        [(args.agent0, args.agent0_ckpt, args.switch_agent_path), (args.agent1, args.agent1_ckpt, args.switch_agent_path)]
    )
    rule_vec = _params_to_vec(battle_params, rule_keys)

    agent0 = _make_side(args.agent0, args.agent0_ckpt, args.switch_agent_path, battle_params, bool(args.stochastic))
    agent1 = _make_side(args.agent1, args.agent1_ckpt, args.switch_agent_path, battle_params, bool(args.stochastic))

    team_gen = load_team_generator(args.team_generator, args.generator_path.strip() or None)
    fm = FixedMatches(
        (agent0, agent1),
        n_team_pairs=int(args.n_team_pairs),
        team_gen=team_gen,
        max_team_size=int(args.max_team_size),
        max_pkm_moves=int(args.max_pkm_moves),
    )
    fm.set_params(battle_params)
    use_predictor = bool(str(args.move_predictor_ckpt).strip())
    fm.run(collect_counts=use_predictor)

    rollout_score = evaluate_rules(fm.rollouts, fm.results)
    print(f"evaluate_rules (rollouts) score={rollout_score:.6f}  (menor = melhor, alvo 20/60/20)")

    if not use_predictor:
        print(f"STAB_MODIFIER={getattr(battle_params, 'STAB_MODIFIER', None)!r}")
        return

    bundle = load_move_predictor(args.move_predictor_ckpt)
    pred_profiles: list[dict[str, float]] = []
    mae_counts: list[float] = []
    kl_sim: list[float] = []

    for ti, (t0, t1) in enumerate(fm.team_pairs):
        for orientation in (0, 1):
            teams = (t0, t1) if orientation == 0 else (t1, t0)
            x = build_x_from_teams(teams[0], teams[1], rule_vec=rule_vec, concat_rules=True)
            pred = predict_move_counts(bundle, x)[0]
            pred_profiles.append(action_profile_from_counts(pred))

            rollout_i = 2 * ti + orientation
            if rollout_i < len(fm.counts_side0):
                # Lado 0 do motor = primeira equipa do par (t0 ou t1 conforme orientação)
                emp = np.asarray(fm.counts_side0[rollout_i], dtype=np.float64)
                mae_counts.append(float(np.abs(pred - emp).mean()))
                if emp.size == 9:
                    p_true = counts_to_prob(emp.reshape(1, -1))
                    q_pred = predict_prob_9(bundle, x.reshape(1, -1))
                    kl_sim.append(float(np.maximum(kl_divergence(p_true, q_pred)[0], 0.0)))

    if mae_counts:
        print(f"predictor vs slot-counts MAE (média)={float(np.mean(mae_counts)):.4f}")
    if kl_sim:
        print(
            f"KL(true||rede) em simulação (média)={float(np.mean(kl_sim)):.4f}  "
            f"mediana={float(np.median(kl_sim)):.4f}  n={len(kl_sim)}"
        )

    combined = evaluate_combined_balance(fm.rollouts, fm.results, pred_profiles)
    print(
        "balance combinado: "
        f"rollout={combined['rollout_rules']:.4f}  "
        f"pred_vs_winner_mae={combined['predictor_vs_winner_mae']:.4f}  "
        f"fast_pred_target={combined['fast_predictor_target']:.4f}  "
        f"combined={combined['combined']:.4f}"
    )
    fast_only = np.mean([balance_score_from_profile(p) for p in pred_profiles])
    print(f"fast predictor-only target distance (média)={fast_only:.4f}")
    print(f"STAB_MODIFIER={getattr(battle_params, 'STAB_MODIFIER', None)!r}")


if __name__ == "__main__":
    main()
