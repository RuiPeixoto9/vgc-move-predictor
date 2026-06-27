"""Dataset de *analytics* de moves (um exemplo por batalha).

Objetivo (reunião 2026-05-04):
- **X**: encoding das equipas iniciais (lado 0 + lado 1) + parâmetros das regras (features).
- **Y/analytics crus**: vencedor e sequência de moves usados ao longo do jogo (por turno e por lado).

O foco aqui NÃO é imitar a política turno-a-turno (isso já existe em `gen_move_dataset.py`).
Aqui queremos um dataset para prever estatísticas/analytics do jogo inteiro sem simular.

Saída:
  `.npz` com:
    - X: (N, 2*team_enc_dim) float32
    - rule_params: (N, K) float32 (K = nº rule_features, na ordem de `rule_feature_keys`)
    - winner: (N,) int64  (0/1)
    - rollout_len: (N,) int64 (nº de turnos)
    - rollout_move_ids: (N, T_max, 2) int64  (pares (lado0, lado1), padding = -999)
    - rollout_move_names_json: (N,) object (JSON list por batalha, para debug/inspeção)
    - move_counts_side0 / move_counts_side1: (N, 2*max_pkm_moves+1) int64 — contagens por slot
      (pokémon por ordem **active+reserve** no início × slots de move) + último canal = **nº de switches**
    - metadados: encoder, seed, team_generator, generator sha256, etc.

Exemplo:
  python -u gen_moves_analytics_dataset.py --episodes 2000 --out moves_analytics_v1.npz \
    --agent_ckpt sup_guidelines_rules.pt --stochastic \
    --max_team_size 2 --max_pkm_moves 4 \
    --team_generator external --generator_path "c:/Users/Rui/Desktop/cenas_uni/estagio/gen.py" \
    --rule_features "STAB_MODIFIER:1.2:1.8,WEATHER_BOOST:1.0:2.0,LIGHT_SCREEN_MODIFIER:0.35:0.65,REFLECT_MODIFIER:0.35:0.65" \
    --resample_rules_each_episode
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vgc2.battle_engine import BattleEngine, BattleRuleParam, State, StateView, TeamView
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.fixed_matches import SWITCH, run_battle_and_slot_counts
from vgc2.ml.external_generators import load_team_generator
from vgc2.ml.battle_policies import make_self_play_pair
from vgc2.ml.neural_policy import inference_battle_policy_from_ckpt, rule_feature_spec_from_ckpt
from vgc2.ml.rule_sampling import parse_rule_feature_ranges, sample_battle_rule_params
from vgc2.util.encoding import EncodeContext, encode_team


PAD_ID = -999


def _sha256_file(path: Path) -> tuple[str, int]:
    if not path.is_file():
        return "", 0
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    try:
        mtime_ns = int(path.stat().st_mtime_ns)
    except OSError:
        mtime_ns = 0
    return h.hexdigest(), mtime_ns


def _infer_team_encoding_size(example_team, ctx: EncodeContext) -> int:
    buf = np.zeros(10000, dtype=np.float32)
    used = encode_team(buf, example_team, ctx)
    return int(used)


def _encode_team_pair(dst: np.ndarray, team_pair, ctx: EncodeContext) -> None:
    t0, t1 = team_pair
    i = encode_team(dst, t0, ctx)
    encode_team(dst[i:], t1, ctx)


def _params_to_feature_vector(params: BattleRuleParam, keys: tuple[str, ...]) -> np.ndarray:
    if not keys:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray([float(getattr(params, k)) for k in keys], dtype=np.float32)


def _move_id_and_name(move) -> tuple[int, str]:
    # move pode ser o SWITCH (id default -1 e name=""), ou um Move real.
    mid = int(getattr(move, "id", -1))
    name = str(getattr(move, "name", "")) or str(move)
    if move is SWITCH or (mid == -1 and (not str(getattr(move, "name", "")))):
        return -1, "SWITCH"
    return mid, name


def _make_engine_and_views(teams, params: BattleRuleParam) -> tuple[BattleEngine, tuple[StateView, StateView], tuple[TeamView, TeamView]]:
    team_view = TeamView(teams[0]), TeamView(teams[1])
    state = State(get_battle_teams(teams, 1))
    state_view = StateView(state, 0, team_view), StateView(state, 1, team_view)
    engine = BattleEngine(state, params)
    return engine, state_view, team_view


def _save_shard(
    out_path: Path,
    *,
    X: np.ndarray,
    rule_mat: np.ndarray,
    winners: np.ndarray,
    rollout_lens: np.ndarray,
    rollout_ids: list[np.ndarray],
    rollout_names_json: np.ndarray,
    move_counts_side0: np.ndarray,
    move_counts_side1: np.ndarray,
    pad_id: int,
    meta: dict,
) -> None:
    t_max = int(max((r.shape[0] for r in rollout_ids), default=0))
    rollout_ids_padded = np.full((int(X.shape[0]), t_max, 2), pad_id, dtype=np.int64)
    for i, r in enumerate(rollout_ids):
        rollout_ids_padded[i, : r.shape[0], :] = r

    np.savez_compressed(
        out_path,
        X=X,
        rule_params=rule_mat,
        winner=winners,
        rollout_len=rollout_lens,
        rollout_move_ids=rollout_ids_padded,
        rollout_move_names_json=rollout_names_json,
        move_counts_side0=move_counts_side0,
        move_counts_side1=move_counts_side1,
        **meta,
        pad_id=int(pad_id),
        kind="moves_analytics",
    )
    print(
        f"Guardado: {out_path.resolve()}\n"
        f"  X={X.shape} rule_params={rule_mat.shape} winner={winners.shape}\n"
        f"  rollout_move_ids={rollout_ids_padded.shape} (T_max={t_max}, pad={pad_id})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera dataset de analytics de moves (um exemplo por batalha).")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--out", type=str, default="moves_analytics_dataset.npz")
    ap.add_argument(
        "--shard_size",
        type=int,
        default=0,
        help="Se >0, escreve múltiplos shards com este tamanho (evita estourar RAM).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--agent_mode",
        type=str,
        default="hybrid_neural_switch",
        help="neural | switch_exploit | tree | random | hybrid_neural_switch | hybrid_switch_greedy",
    )
    ap.add_argument(
        "--agent_ckpt",
        type=str,
        default="sup_guidelines_rules.pt",
        help="Checkpoint IL (obrigatório para neural e hybrid_neural_switch).",
    )
    ap.add_argument(
        "--switch_agent_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "battle_agent.py"),
        help="battle_agent.py com SwitchExploitPolicy.",
    )
    ap.add_argument("--tree_depth", type=int, default=2)
    ap.add_argument("--random_switch_prob", type=float, default=0.15)
    ap.add_argument("--hybrid_switch_prob", type=float, default=0.5)
    ap.add_argument("--stochastic", action="store_true", help="Rede em modo estocástico (melhor para dataset).")
    ap.add_argument("--max_team_size", type=int, default=2)
    ap.add_argument("--max_pkm_moves", type=int, default=4)
    ap.add_argument("--team_generator", choices=("default", "external"), default="default")
    ap.add_argument("--generator_path", type=str, default="")
    ap.add_argument("--rule_features", type=str, default="")
    ap.add_argument("--resample_rules_each_episode", action="store_true")
    args = ap.parse_args()

    np.random.seed(int(args.seed))

    agent_mode = str(args.agent_mode).strip().lower()
    ckpt_path = str(args.agent_ckpt).strip()
    agent0, agent1, policy_meta = make_self_play_pair(
        agent_mode,
        agent_ckpt=ckpt_path,
        switch_agent_path=str(args.switch_agent_path).strip(),
        stochastic=bool(args.stochastic),
        tree_depth=int(args.tree_depth),
        random_switch_prob=float(args.random_switch_prob),
        hybrid_switch_prob=float(args.hybrid_switch_prob),
    )

    ckpt = None
    if ckpt_path and agent_mode in ("neural", "hybrid_neural_switch"):
        _, ckpt = inference_battle_policy_from_ckpt(ckpt_path, stochastic=bool(args.stochastic))

    # Rule spec: se o checkpoint tiver, usa o dele (consistência); senão, usa CLI.
    ck_keys, ck_ranges = rule_feature_spec_from_ckpt(ckpt) if ckpt is not None else ([], {})
    if ck_keys:
        rule_feature_keys: tuple[str, ...] = tuple(ck_keys)
        rule_feature_ranges = dict(ck_ranges)
    else:
        rule_feature_keys, rule_feature_ranges = parse_rule_feature_ranges(args.rule_features)

    team_gen = load_team_generator(args.team_generator, args.generator_path.strip() or None)

    # Infer encoding size a partir de uma equipa gerada
    ctx = EncodeContext()
    example_team = team_gen(int(args.max_team_size), int(args.max_pkm_moves))
    team_dim = _infer_team_encoding_size(example_team, ctx)
    x_dim = 2 * team_dim

    move_count_dim = int(args.max_team_size) * int(args.max_pkm_moves) + 1

    X = np.zeros((int(args.episodes), x_dim), dtype=np.float32)
    winners = np.zeros((int(args.episodes),), dtype=np.int64)
    rollout_lens = np.zeros((int(args.episodes),), dtype=np.int64)
    rule_mat = np.zeros((int(args.episodes), len(rule_feature_keys)), dtype=np.float32)
    move_counts_side0 = np.zeros((int(args.episodes), move_count_dim), dtype=np.int64)
    move_counts_side1 = np.zeros((int(args.episodes), move_count_dim), dtype=np.int64)

    # guardamos rollout ids e nomes em listas e só no fim fazemos padding
    rollout_ids: list[np.ndarray] = []
    rollout_names_json = np.empty((int(args.episodes),), dtype=object)

    gen_path = Path(args.generator_path).resolve() if args.generator_path else None
    gen_sha, gen_mtime = _sha256_file(gen_path) if gen_path else ("", 0)
    ck_path = Path(ckpt_path).resolve() if ckpt_path else None
    ck_sha, ck_mtime = _sha256_file(ck_path) if ck_path and ck_path.is_file() else ("", 0)

    meta = dict(
        seed=int(args.seed),
        max_team_size=int(args.max_team_size),
        max_pkm_moves=int(args.max_pkm_moves),
        team_encoder_dim=int(team_dim),
        team_generator=str(args.team_generator),
        generator_path=str(gen_path) if gen_path else "",
        generator_sha256=gen_sha,
        generator_mtime_ns=gen_mtime,
        agent_ckpt=str(ck_path) if ck_path else "",
        agent_ckpt_sha256=ck_sha,
        agent_ckpt_mtime_ns=ck_mtime,
        stochastic=bool(args.stochastic),
        rule_feature_keys=tuple(rule_feature_keys),
        rule_feature_ranges=dict(rule_feature_ranges),
        rule_feature_ranges_json=np.array(
            json.dumps(
                {k: [float(rule_feature_ranges[k][0]), float(rule_feature_ranges[k][1])] for k in rule_feature_keys}
            )
            if rule_feature_keys
            else "",
            dtype=object,
        ),
        resample_rules_each_episode=bool(args.resample_rules_each_episode),
        move_count_dim=int(move_count_dim),
        move_count_layout=np.array(
            "slots[pkm_i * max_pkm_moves + slot_j] for pkm_i in order(active+reserve at battle start); "
            "last index = switch count",
            dtype=object,
        ),
    )
    meta.update(policy_meta)
    meta["agent_ckpt_sha256"] = ck_sha
    meta["agent_ckpt_mtime_ns"] = ck_mtime

    print(
        "Coleta: "
        f"episodes={args.episodes} mode={agent_mode} ckpt={ckpt_path or '(n/a)'} "
        f"stochastic={bool(args.stochastic)} team_generator={args.team_generator} "
        f"rule_keys={rule_feature_keys or '(none)'}"
    )

    shard_size = int(args.shard_size) if int(args.shard_size) > 0 else int(args.episodes)
    out_path = Path(args.out)
    base = out_path
    suffix = out_path.suffix.lower()
    if int(args.shard_size) > 0:
        # `--out foo.npz` => shards `foo_part0000.npz`, etc.
        if suffix != ".npz":
            raise SystemExit("--out deve terminar em .npz quando usas --shard_size")
        base = out_path.with_suffix("")

    shard_idx = 0
    shard_start = 0

    for ep in range(int(args.episodes)):
        teams = (team_gen(int(args.max_team_size), int(args.max_pkm_moves)),
                 team_gen(int(args.max_team_size), int(args.max_pkm_moves)))

        params: BattleRuleParam
        if rule_feature_keys and (bool(args.resample_rules_each_episode) or ep == 0):
            params = sample_battle_rule_params(rule_feature_keys, rule_feature_ranges)
        else:
            params = BattleRuleParam()

        # políticas precisam das regras
        agent0.set_params(params)
        agent1.set_params(params)

        engine, views, _ = _make_engine_and_views(teams, params)
        winner, rollout, c0, c1 = run_battle_and_slot_counts(engine, (agent0, agent1), views)
        if int(c0.shape[0]) != move_count_dim or int(c1.shape[0]) != move_count_dim:
            raise RuntimeError(
                f"Dimensão de contagens inesperada: side0={c0.shape} side1={c1.shape} "
                f"(esperado {move_count_dim}=max_team_size*max_pkm_moves+1)"
            )

        _encode_team_pair(X[ep], teams, ctx)
        winners[ep] = int(winner)
        rule_mat[ep] = _params_to_feature_vector(params, rule_feature_keys)
        move_counts_side0[ep] = c0
        move_counts_side1[ep] = c1

        ids_ep = np.zeros((len(rollout), 2), dtype=np.int64)
        names_ep: list[list[str]] = []
        for t, (m0, m1) in enumerate(rollout):
            id0, n0 = _move_id_and_name(m0)
            id1, n1 = _move_id_and_name(m1)
            ids_ep[t, 0] = id0
            ids_ep[t, 1] = id1
            names_ep.append([n0, n1])

        rollout_ids.append(ids_ep)
        rollout_lens[ep] = int(ids_ep.shape[0])
        rollout_names_json[ep] = json.dumps(names_ep, ensure_ascii=False)

        if (ep + 1) % max(1, int(args.episodes) // 20) == 0:
            # evita caracteres fora de cp1252 no Windows
            print(f"  ep {ep + 1}/{args.episodes}  turns_mean~{float(np.mean(rollout_lens[: ep + 1])):.1f}")

        # flush de shard (se ativado)
        if (ep + 1) % shard_size == 0:
            n = (ep + 1) - shard_start
            if int(args.shard_size) > 0:
                shard_out = Path(f"{str(base)}_part{shard_idx:04d}.npz")
            else:
                shard_out = out_path
            _save_shard(
                shard_out,
                X=X[shard_start : shard_start + n],
                rule_mat=rule_mat[shard_start : shard_start + n],
                winners=winners[shard_start : shard_start + n],
                rollout_lens=rollout_lens[shard_start : shard_start + n],
                rollout_ids=rollout_ids[shard_start : shard_start + n],
                rollout_names_json=rollout_names_json[shard_start : shard_start + n],
                move_counts_side0=move_counts_side0[shard_start : shard_start + n],
                move_counts_side1=move_counts_side1[shard_start : shard_start + n],
                pad_id=PAD_ID,
                meta=meta | {"shard_index": int(shard_idx), "shard_start": int(shard_start), "shard_size": int(n)},
            )
            shard_idx += 1
            shard_start = ep + 1

    # resto (último shard parcial)
    if shard_start < int(args.episodes):
        n = int(args.episodes) - shard_start
        if int(args.shard_size) > 0:
            shard_out = Path(f"{str(base)}_part{shard_idx:04d}.npz")
        else:
            shard_out = out_path
        _save_shard(
            shard_out,
            X=X[shard_start : shard_start + n],
            rule_mat=rule_mat[shard_start : shard_start + n],
            winners=winners[shard_start : shard_start + n],
            rollout_lens=rollout_lens[shard_start : shard_start + n],
            rollout_ids=rollout_ids[shard_start : shard_start + n],
            rollout_names_json=rollout_names_json[shard_start : shard_start + n],
            move_counts_side0=move_counts_side0[shard_start : shard_start + n],
            move_counts_side1=move_counts_side1[shard_start : shard_start + n],
            pad_id=PAD_ID,
            meta=meta | {"shard_index": int(shard_idx), "shard_start": int(shard_start), "shard_size": int(n)},
        )


if __name__ == "__main__":
    main()

