"""Balancer simples: GA (pygad) optimiza EVs de 1 equipa vs target de usage (previsor).

Ideia (orientador, 2026-05):
  - Fitness = KL(target || previsor) — **sem simular batalhas** no loop.
  - Genes = pesos de EV (2 pokémon × 6 stats) → normalizados a 510 EVs/mon.
  - Equipa adversária fixa; regras fixas.
  - No fim: simular 1 batalha com o melhor indivíduo e comparar previsão vs empírico.

Requer: pip install pygad  (ou usa --engine numpy)

Exemplo:
  python -u balance_team_predictor_ga.py \\
    --ckpt move_predictor_local.pt \\
    --target_switch 0.25 \\
    --team_generator external --generator_path ../gen.py \\
    --generations 40 --pop_size 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from eval_move_predictor_kl import counts_to_prob, kl_divergence, predict_prob_9
from move_predictor_io import build_x_from_teams, load_move_predictor
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam, State, StateView, Team, TeamView
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.fixed_matches import run_battle_and_slot_counts
from vgc2.ml.battle_policies import make_self_play_pair
from vgc2.ml.external_generators import load_team_generator
from vgc2.ml.neural_policy import rule_feature_spec_from_ckpt
from vgc2.ml.rule_sampling import sample_battle_rule_params


N_GENES = 12  # 2 mon × 6 stats


def _evs_from_weights(w: np.ndarray) -> tuple[int, ...]:
    w = np.clip(np.asarray(w, dtype=np.float64), 1e-6, None)
    w = w / w.sum()
    raw = w * 510.0
    ev = np.floor(raw).astype(int)
    # Corrigir soma para 510
    diff = 510 - int(ev.sum())
    for _ in range(abs(diff)):
        idx = int(np.argmax(w if diff > 0 else -w))
        ev[idx] += 1 if diff > 0 else -1
    ev = np.clip(ev, 0, 252)
    s = int(ev.sum())
    if s != 510:
        ev[0] += 510 - s
    return tuple(int(x) for x in ev)


def _team_from_genes(base: Team, genes: np.ndarray) -> Team:
    g = np.asarray(genes, dtype=np.float64).reshape(N_GENES)
    members = []
    for mi in range(min(2, len(base.members))):
        base_p = base.members[mi]
        w = g[mi * 6 : (mi + 1) * 6]
        evs = _evs_from_weights(w)
        members.append(
            type(base_p)(
                species=base_p.species,
                move_indexes=list(base_p._move_indexes),
                level=base_p.level,
                evs=evs,
                ivs=base_p.ivs,
                nature=base_p.nature,
            )
        )
    return Team(members)


def _default_target_9(*, switch: float = 0.20) -> np.ndarray:
    """8 slots de move uniformes + switch desejado (soma 1)."""
    sw = float(np.clip(switch, 0.0, 0.9))
    move_mass = 1.0 - sw
    t = np.full(9, move_mass / 8.0, dtype=np.float64)
    t[8] = sw
    return t / t.sum()


def _parse_target_9(s: str) -> np.ndarray:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 9:
        raise ValueError("target9 precisa de 9 valores separados por vírgula")
    a = np.asarray(parts, dtype=np.float64)
    a = np.clip(a, 0.0, None)
    return a / a.sum()


def _battle_params_and_rules(agent_ckpt: str) -> tuple[BattleRuleParam, tuple[str, ...], np.ndarray]:
    ck = torch.load(str(agent_ckpt), map_location="cpu")
    keys, ranges = rule_feature_spec_from_ckpt(ck)
    if not keys:
        return BattleRuleParam(), (), np.zeros((0,), dtype=np.float32)
    params = sample_battle_rule_params(keys, ranges)
    rule_vec = np.asarray([float(getattr(params, k)) for k in keys], dtype=np.float32)
    return params, keys, rule_vec


def _kl_predictor_vs_target(
    bundle: dict,
    team0: Team,
    team1: Team,
    rule_vec: np.ndarray,
    target: np.ndarray,
    *,
    eps: float,
) -> tuple[float, np.ndarray]:
    x = build_x_from_teams(team0, team1, rule_vec=rule_vec, concat_rules=True)
    q = predict_prob_9(bundle, x.reshape(1, -1), eps=eps)[0]
    p = np.asarray(target, dtype=np.float64)
    kl = float(np.maximum(kl_divergence(p.reshape(1, -1), q.reshape(1, -1), eps=eps)[0], 0.0))
    return kl, q


def _simulate_side0_counts(
    team0: Team,
    team1: Team,
    params: BattleRuleParam,
    *,
    agent_ckpt: str,
    switch_path: str,
) -> np.ndarray:
    agent0, _, _ = make_self_play_pair(
        "hybrid_neural_switch",
        agent_ckpt=agent_ckpt,
        switch_agent_path=switch_path,
        stochastic=False,
    )
    agent1 = GreedyBattlePolicy()
    agent0.set_params(params)
    agent1.set_params(params)
    teams = (team0, team1)
    team_view = TeamView(teams[0]), TeamView(teams[1])
    state = State(get_battle_teams(teams, 1))
    views = StateView(state, 0, team_view), StateView(state, 1, team_view)
    engine = BattleEngine(state, params)
    _, _, counts0, _ = run_battle_and_slot_counts(engine, (agent0, agent1), views)
    return np.asarray(counts0, dtype=np.float64)


def _run_pygad(
    fitness_fn,
    *,
    pop_size: int,
    generations: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    import pygad

    ga = pygad.GA(
        num_generations=int(generations),
        num_parents_mating=max(2, int(pop_size // 5)),
        fitness_func=fitness_fn,
        sol_per_pop=int(pop_size),
        num_genes=N_GENES,
        gene_space=[{"low": 0.01, "high": 1.0}] * N_GENES,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=15,
        random_seed=int(seed),
        save_best_solutions=True,
    )
    ga.run()
    solution, fitness, _ = ga.best_solution()
    return np.asarray(solution, dtype=np.float64), float(fitness)


def _run_numpy_ga(
    fitness_fn,
    *,
    pop_size: int,
    generations: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    pop = rng.uniform(0.01, 1.0, size=(pop_size, N_GENES))
    best_g, best_f = pop[0], -np.inf

    for _ in range(int(generations)):
        scores = np.array([fitness_fn(None, g, 0) for g in pop])
        order = np.argsort(scores)[::-1]
        pop = pop[order]
        scores = scores[order]
        if scores[0] > best_f:
            best_f, best_g = float(scores[0]), pop[0].copy()

        n_elite = max(2, pop_size // 10)
        parents = pop[: max(4, pop_size // 3)]
        children: list[np.ndarray] = [pop[i].copy() for i in range(n_elite)]
        while len(children) < pop_size:
            p0, p1 = parents[rng.integers(0, len(parents), size=2)]
            cut = rng.integers(1, N_GENES)
            child = np.concatenate([p0[:cut], p1[cut:]])
            mut = rng.random(N_GENES) < 0.15
            child[mut] = rng.uniform(0.01, 1.0, size=int(mut.sum()))
            children.append(child)
        pop = np.stack(children[:pop_size], axis=0)

    return best_g, best_f


def main() -> None:
    ap = argparse.ArgumentParser(description="GA: optimiza EVs de 1 equipa via previsor (sem sim no loop).")
    ap.add_argument("--ckpt", type=str, default="move_predictor_local.pt")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--agent_ckpt", type=str, default="sup_guidelines_rules.pt")
    ap.add_argument(
        "--switch_agent_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "battle_agent.py"),
    )
    ap.add_argument("--team_generator", choices=("default", "external"), default="external")
    ap.add_argument("--generator_path", type=str, default="../gen.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_switch", type=float, default=0.25, help="Massa alvo no slot SW (resto uniforme nos 8 moves).")
    ap.add_argument("--target9", type=str, default="", help="9 probs custom separadas por vírgula (override target_switch).")
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--pop_size", type=int, default=24)
    ap.add_argument("--engine", choices=("pygad", "numpy", "auto"), default="auto")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--skip_sim", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rng = np.random.default_rng(int(args.seed))
    target = _parse_target_9(args.target9) if args.target9.strip() else _default_target_9(switch=args.target_switch)

    params, _, rule_vec = _battle_params_and_rules(args.agent_ckpt)
    team_gen = load_team_generator(args.team_generator, args.generator_path.strip() or None)
    base0 = team_gen(2, 4)
    team1 = team_gen(2, 4)

    bundle = load_move_predictor(args.ckpt, args.device)

    base_kl, base_q = _kl_predictor_vs_target(bundle, base0, team1, rule_vec, target, eps=args.eps)
    print(f"baseline KL(target||pred)={base_kl:.4f}  target_switch={target[8]:.3f}")

    def fitness(_ga, solution, _idx):
        t0 = _team_from_genes(base0, solution)
        kl, _ = _kl_predictor_vs_target(bundle, t0, team1, rule_vec, target, eps=args.eps)
        return -kl  # maximizar

    engine = args.engine
    if engine == "auto":
        try:
            import pygad  # noqa: F401

            engine = "pygad"
        except ImportError:
            engine = "numpy"
            print("AVISO: pygad não instalado — a usar GA numpy.  pip install pygad")

    if engine == "pygad":
        best_genes, best_fit = _run_pygad(
            fitness, pop_size=args.pop_size, generations=args.generations, seed=args.seed
        )
    else:
        best_genes, best_fit = _run_numpy_ga(
            fitness, pop_size=args.pop_size, generations=args.generations, seed=args.seed
        )

    best_team = _team_from_genes(base0, best_genes)
    best_kl, best_q = _kl_predictor_vs_target(bundle, best_team, team1, rule_vec, target, eps=args.eps)
    print(f"best KL(target||pred)={best_kl:.4f}  (fitness={best_fit:.4f})  engine={engine}")
    print(f"  pred SW={best_q[8]:.3f}  target SW={target[8]:.3f}")

    if args.skip_sim:
        return

    emp = _simulate_side0_counts(
        best_team,
        team1,
        params,
        agent_ckpt=args.agent_ckpt,
        switch_path=args.switch_agent_path,
    )
    emp_p = counts_to_prob(emp.reshape(1, -1), eps=args.eps)[0]
    kl_emp_pred = float(
        np.maximum(kl_divergence(emp_p.reshape(1, -1), best_q.reshape(1, -1), eps=args.eps)[0], 0.0)
    )
    kl_emp_target = float(
        np.maximum(kl_divergence(target.reshape(1, -1), emp_p.reshape(1, -1), eps=args.eps)[0], 0.0)
    )
    print(
        f"pós-sim (1 batalha): emp SW={emp_p[8]:.3f}  "
        f"KL(emp||pred)={kl_emp_pred:.4f}  KL(target||emp)={kl_emp_target:.4f}"
    )
    print(
        "  (KL target||emp mede se a simulação bate a target; "
        "KL emp||pred mede erro do previsor na equipa optimizada)"
    )


if __name__ == "__main__":
    main()
