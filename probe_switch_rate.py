"""Mede taxa de switches e uso do 2.º pokémon em N batalhas (diagnóstico)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vgc2.battle_engine import BattleRuleParam, State, StateView, TeamView
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.fixed_matches import run_battle_and_slot_counts
from vgc2.ml.battle_policies import make_self_play_pair
from vgc2.ml.external_generators import load_team_generator
from vgc2.ml.rule_sampling import parse_rule_feature_ranges, sample_battle_rule_params


def _make_engine_and_views(teams, params):
    team_view = TeamView(teams[0]), TeamView(teams[1])
    state = State(get_battle_teams(teams, 1))
    views = StateView(state, 0, team_view), StateView(state, 1, team_view)
    from vgc2.battle_engine import BattleEngine

    return BattleEngine(state, params), views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=500)
    ap.add_argument("--agent_mode", type=str, default="hybrid_neural_switch")
    ap.add_argument("--agent_ckpt", type=str, default="sup_guidelines_rules.pt")
    ap.add_argument(
        "--switch_agent_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "battle_agent.py"),
    )
    ap.add_argument("--max_team_size", type=int, default=2)
    ap.add_argument("--max_pkm_moves", type=int, default=4)
    ap.add_argument("--team_generator", choices=("default", "external"), default="external")
    ap.add_argument("--generator_path", type=str, default="")
    ap.add_argument("--rule_features", type=str, default="")
    ap.add_argument("--tree_depth", type=int, default=2)
    args = ap.parse_args()

    team_gen = load_team_generator(args.team_generator, args.generator_path.strip() or None)
    rule_keys, rule_ranges = parse_rule_feature_ranges(args.rule_features)

    agent0, agent1, _ = make_self_play_pair(
        args.agent_mode,
        agent_ckpt=args.agent_ckpt,
        switch_agent_path=args.switch_agent_path,
        tree_depth=args.tree_depth,
    )

    n_sw0 = n_sw1 = 0
    n_pkm1_use0 = n_pkm1_use1 = 0
    n_b = int(args.battles)

    for _ in range(n_b):
        teams = (
            team_gen(int(args.max_team_size), int(args.max_pkm_moves)),
            team_gen(int(args.max_team_size), int(args.max_pkm_moves)),
        )
        params = sample_battle_rule_params(rule_keys, rule_ranges) if rule_keys else BattleRuleParam()
        agent0.set_params(params)
        agent1.set_params(params)
        engine, views = _make_engine_and_views(teams, params)
        _, _, c0, c1 = run_battle_and_slot_counts(engine, (agent0, agent1), views)
        if c0[8] > 0:
            n_sw0 += 1
        if c1[8] > 0:
            n_sw1 += 1
        if c0[4:8].sum() > 0:
            n_pkm1_use0 += 1
        if c1[4:8].sum() > 0:
            n_pkm1_use1 += 1

    print(f"mode={args.agent_mode}  battles={n_b}")
    print(f"  P(switch side0)>0: {n_sw0/n_b:.3f}  P(switch side1)>0: {n_sw1/n_b:.3f}")
    print(f"  P(pkm1 slots used side0): {n_pkm1_use0/n_b:.3f}  side1: {n_pkm1_use1/n_b:.3f}")


if __name__ == "__main__":
    main()
