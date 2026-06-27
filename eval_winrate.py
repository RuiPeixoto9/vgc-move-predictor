"""Taxa de vitória do checkpoint (lado 0) vs greedy ou random (lado 1)."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from vgc2.agent.battle import GreedyBattlePolicy, RandomBattlePolicy
from vgc2.battle_engine import BattleEngine, State, StateView, TeamView
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.match import label_teams, run_battle
from vgc2.ml.env import BattleEnv
from vgc2.ml.external_generators import load_team_generator
from vgc2.ml.neural_policy import inference_battle_policy_from_ckpt, rule_feature_spec_from_ckpt
from vgc2.ml.rule_sampling import sample_battle_rule_params


def evaluate_checkpoint(
    ckpt_path: str,
    games: int,
    opponent: str = "greedy",
    seed: int = 0,
    *,
    stochastic: bool = False,
    progress: bool = True,
    team_generator: str = "default",
    generator_path: str = "",
) -> tuple[int, int]:
    """Corre `games` batalhas com a mesma sequência RNG que `np.random.seed(seed)` fixa antes do loop."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    neural, ckpt = inference_battle_policy_from_ckpt(
        ckpt_path, stochastic=stochastic
    )
    n_active = int(ckpt.get("n_active", 2))
    max_team_size = int(ckpt.get("max_team_size", 4))
    max_pkm_moves = int(ckpt.get("max_pkm_moves", 4))
    encoder = str(ckpt.get("encoder", "full"))
    team_gen = load_team_generator(team_generator, generator_path.strip() or None)
    env = BattleEnv(
        encoder=encoder,
        n_active=n_active,
        max_team_size=max_team_size,
        max_pkm_moves=max_pkm_moves,
        _gen_team=team_gen,
    )
    params = env.params

    rule_feature_keys, rule_feature_ranges = rule_feature_spec_from_ckpt(ckpt)

    if opponent == "greedy":
        opp = GreedyBattlePolicy()
        opp.params = params
    else:
        opp = RandomBattlePolicy()

    wins = 0
    for g in range(games):
        team = team_gen(max_team_size, max_pkm_moves), team_gen(max_team_size, max_pkm_moves)
        label_teams(team)
        tv = TeamView(team[0]), TeamView(team[1])
        state = State(get_battle_teams(team, n_active))
        sv = StateView(state, 0, tv), StateView(state, 1, tv)
        if rule_feature_keys:
            battle_params = sample_battle_rule_params(rule_feature_keys, rule_feature_ranges)
            neural.set_params(battle_params)
            if opponent == "greedy":
                opp.set_params(battle_params)
            engine = BattleEngine(state, battle_params, debug=False)
        else:
            engine = BattleEngine(state, debug=False)
        winner = run_battle(engine, (neural, opp), tv, sv, None)
        if winner == 0:
            wins += 1
        if progress and (g + 1) % max(1, games // 10) == 0:
            print(f"  {g + 1}/{games}  vitórias={wins}  ({100.0 * wins / (g + 1):.1f}%)")

    return wins, games


def main() -> None:
    p = argparse.ArgumentParser(description="Win rate: rede (lado 0) vs oponente.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--opponent", choices=("greedy", "random"), default="greedy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true", help="Amostragem na rede (default: determinístico).")
    p.add_argument("--team_generator", choices=("default", "external"), default="default")
    p.add_argument("--generator_path", type=str, default="")
    args = p.parse_args()

    wins, games = evaluate_checkpoint(
        args.ckpt,
        args.games,
        args.opponent,
        args.seed,
        stochastic=args.stochastic,
        progress=True,
        team_generator=args.team_generator,
        generator_path=args.generator_path,
    )
    rate = wins / games
    print(
        f"\n{args.ckpt}  vs {args.opponent}  n={games}  "
        f"wins_side0={wins}  win_rate={rate:.3f}"
    )


if __name__ == "__main__":
    main()
