"""Demonstração visual no Godot com o agente treinado (IL + switch híbrido).

Pré-requisitos:
  1. Godot 4.4 aberto com o projeto `visual_server/` (botão Play / F5).
     Deve aparecer no output do Godot: "UDP listening on port 12345".
  2. Este script a correr (envia eventos UDP para o Godot animar).

Exemplo (Git Bash):
  cd /c/Users/Rui/Desktop/cenas_uni/estagio/pokemon-vgc-engine
  source .venv/Scripts/activate
  python -u demo_godot_battle.py \\
    --agent_ckpt sup_guidelines_rules.pt \\
    --switch_agent_path ../battle_agent.py \\
    --team_generator external --generator_path ../gen.py

Gravar vídeo: captura a janela do Godot (OBS, Win+G, etc.) enquanto o script corre.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam, State, StateView, TeamView
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.match import label_teams, run_battle
from vgc2.ml.battle_policies import make_self_play_pair
from vgc2.ml.external_generators import load_team_generator
from vgc2.ml.neural_policy import rule_feature_spec_from_ckpt
from vgc2.ml.rule_sampling import sample_battle_rule_params
from vgc2.net.stream import CLIENT_MAP

import torch


def _load_rules(agent_ckpt: str) -> BattleRuleParam:
    ck = torch.load(str(agent_ckpt), map_location="cpu")
    keys, ranges = rule_feature_spec_from_ckpt(ck)
    if not keys:
        return BattleRuleParam()
    return sample_battle_rule_params(keys, ranges)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batalha visual no Godot com agente neural.")
    ap.add_argument("--agent_mode", default="hybrid_neural_switch")
    ap.add_argument("--agent_ckpt", default="sup_guidelines_rules.pt")
    ap.add_argument("--switch_agent_path", default="../battle_agent.py")
    ap.add_argument("--opponent", choices=("greedy", "neural"), default="greedy")
    ap.add_argument("--team_generator", choices=("default", "external"), default="external")
    ap.add_argument("--generator_path", default="../gen.py")
    ap.add_argument("--max_team_size", type=int, default=2)
    ap.add_argument("--max_pkm_moves", type=int, default=4)
    ap.add_argument("--n_active", type=int, default=1)
    ap.add_argument("--stream", choices=tuple(CLIENT_MAP.keys()), default="godot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stream == "godot":
        print("Certifica-te que o Godot (visual_server) está a correr com Play/F5.")
        print("Porta UDP esperada: 127.0.0.1:12345\n")

    team_gen = load_team_generator(args.team_generator, args.generator_path)
    agent0, agent1_meta, meta = make_self_play_pair(
        args.agent_mode,
        agent_ckpt=args.agent_ckpt,
        switch_agent_path=args.switch_agent_path,
        stochastic=False,
    )
    if args.opponent == "greedy":
        agent1 = GreedyBattlePolicy()
    else:
        agent1, _, _ = make_self_play_pair(
            args.agent_mode,
            agent_ckpt=args.agent_ckpt,
            switch_agent_path=args.switch_agent_path,
            stochastic=False,
        )

    params = _load_rules(args.agent_ckpt)
    agent0.set_params(params)
    agent1.set_params(params)

    team0 = team_gen(args.max_team_size, args.max_pkm_moves)
    team1 = team_gen(args.max_team_size, args.max_pkm_moves)
    teams = (team0, team1)
    label_teams(teams)
    team_view = TeamView(teams[0]), TeamView(teams[1])
    state = State(get_battle_teams(teams, args.n_active))
    views = StateView(state, 0, team_view), StateView(state, 1, team_view)

    client = CLIENT_MAP[args.stream]()
    engine = BattleEngine(state, params, debug=True)

    print("~ Equipa 0 (agente IL+híbrido) ~")
    print(teams[0])
    print("~ Equipa 1 (oponente) ~")
    print(teams[1])
    print(f"Agente: {meta.get('agent_mode')} | ckpt: {meta.get('agent_ckpt', args.agent_ckpt)}")
    print(f"Regras: STAB={getattr(params, 'STAB_MODIFIER', None)}")
    print("A iniciar batalha...\n")

    client.start_stream("demo_battle")
    winner = run_battle(engine, (agent0, agent1), team_view, views, client)
    client.close()
    print(f"Vencedor: jogador {winner}")


if __name__ == "__main__":
    try:
        main()
    except ConnectionRefusedError:
        print("Erro de ligação. O Godot está aberto com visual_server em execução?", file=sys.stderr)
        raise
