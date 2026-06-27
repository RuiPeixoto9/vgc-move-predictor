"""Treino supervisionado do actor (e opcionalmente do critic) por imitação."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import amp as torch_amp
from torch.optim import Adam

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.ml.env import BattleEnv
from vgc2.ml.heuristic import critic_target_after_step
from vgc2.ml.models import BattleActionSpaceSpec, BattleActorCritic
from vgc2.ml.neural_policy import commands_to_gym_action
from vgc2.ml.rule_sampling import rule_features_cli_string_from_npz


def collect_transitions_online(env: BattleEnv, expert: GreedyBattlePolicy, n_transitions: int, with_values: bool):
    obs_list, act_list, v_list = [], [], []
    obs, _ = env.reset()
    while len(obs_list) < n_transitions:
        obs_decision = np.array(obs, dtype=np.float32, copy=True)
        cmds = expert.decision(env.state_view[0], env.team_view[1])
        action = commands_to_gym_action(cmds, env.n_active)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        assert env.engine is not None
        if with_values:
            v_list.append(critic_target_after_step(env.engine, terminated=done))
        obs_list.append(obs_decision)
        act_list.append(action)
        if done:
            obs, _ = env.reset()
    out = (np.stack(obs_list), np.stack(act_list))
    if with_values:
        out = out + (np.array(v_list, dtype=np.float32),)
    return out


def parse_hidden_dims(text: str, hidden_default: int) -> tuple[int, ...]:
    if not text.strip():
        return (hidden_default, hidden_default, hidden_default)
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("hidden_dims vazio")
    return vals


def parse_rule_feature_ranges(text: str) -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    """Formato: KEY:MIN:MAX,KEY2:MIN:MAX"""
    spec = (text or "").strip()
    if not spec:
        return (), {}
    keys: list[str] = []
    ranges: dict[str, tuple[float, float]] = {}
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        bits = [x.strip() for x in part.split(":")]
        if len(bits) != 3:
            raise ValueError(f"rule_features inválido: '{part}'. Use KEY:MIN:MAX")
        key = bits[0]
        lo = float(bits[1])
        hi = float(bits[2])
        if hi <= lo:
            raise ValueError(f"rule_features range inválido em '{part}'")
        keys.append(key)
        ranges[key] = (lo, hi)
    return tuple(keys), ranges


def load_compatible_weights(model: BattleActorCritic, ckpt_state: dict) -> tuple[int, int, int]:
    """Carrega pesos compatíveis e faz warm-start parcial quando obs_dim aumenta.

    Caso típico: adicionamos rule-features ao input, logo a primeira Linear muda de
    [out, old_in] para [out, new_in]. Nessa situação copiamos as colunas antigas e
    deixamos as novas colunas como estavam (init do modelo atual).
    """
    current = model.state_dict()
    compatible = {}
    partial = 0
    for k, v in ckpt_state.items():
        if k not in current:
            continue
        cur = current[k]
        if cur.shape == v.shape:
            compatible[k] = v
            continue
        # Warm-start parcial para pesos de Linear quando só cresce a dimensão de input.
        if (
            cur.ndim == 2
            and v.ndim == 2
            and cur.shape[0] == v.shape[0]
            and cur.shape[1] >= v.shape[1]
        ):
            merged = cur.clone()
            merged[:, : v.shape[1]] = v
            compatible[k] = merged
            partial += 1
    current.update(compatible)
    model.load_state_dict(current, strict=False)
    return len(compatible), len(current) - len(compatible), partial


def pick_training_device(name: str) -> torch.device:
    """Resolve cpu | cuda | cuda:N | mps | auto (CUDA > MPS > CPU)."""
    key = (name or "auto").strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def save_checkpoint(
    path: Path,
    model: BattleActorCritic,
    spec: BattleActionSpaceSpec,
    env: BattleEnv,
    *,
    hidden_dims: tuple[int, ...],
    w_value: float,
) -> None:
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state": state_cpu,
            "obs_dim": env.observation_space.shape[0],
            "action_nvec": spec.nvec,
            "action_start": spec.start,
            "hidden_dim": hidden_dims[0],
            "hidden_dims": hidden_dims,
            "n_active": env.n_active,
            "max_team_size": env.max_team_size,
            "max_pkm_moves": env.max_pkm_moves,
            "w_value": w_value,
            "encoder": env.encoder,
            "use_critic_sigmoid": model.use_critic_sigmoid,
            "rule_feature_keys": tuple(env.rule_feature_keys),
            "rule_feature_ranges": dict(env.rule_feature_ranges),
            "rule_conditioning": model.rule_conditioning,
            "rule_condition_dim": model.rule_condition_dim,
            "film_hidden_dim": model.film_hidden_dim,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Treino supervisionado actor (+ critic opcional).")
    parser.add_argument("--steps", type=int, default=5000, help="Transições por época (modo online).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="",
        help="Lista separada por vírgulas, ex: 512,512,256 (sobrepõe --hidden).",
    )
    parser.add_argument("--out", type=str, default="supervised_actor.pt")
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="Checkpoint inicial opcional para finetuning supervisionado.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder", choices=("full", "compact", "compact_no_opp_reserve", "greedy_subset"), default="compact")
    parser.add_argument("--n_active", type=int, default=2)
    parser.add_argument("--max_team_size", type=int, default=4)
    parser.add_argument("--max_pkm_moves", type=int, default=4)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="PATH",
        help="Ficheiro .npz de gen_transition_dataset.py (obs, actions, critic_target). "
        "Repete o flag para concatenar vários (ex.: dados greedy + ronda DAgger).",
    )
    parser.add_argument(
        "--w_value",
        type=float,
        default=0.5,
        help="Peso L_C na loss total L = L_P + w_value * L_C. Use 0 para só actor.",
    )
    parser.add_argument(
        "--train_critic_online",
        action="store_true",
        help="Sem --dataset: recolhe critic_target em tempo real (mais lento).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Com --dataset: mini-batches com shuffle por época (0 = um batch com tudo).",
    )
    parser.add_argument(
        "--save_every_epochs",
        type=int,
        default=0,
        help="Se >0, guarda checkpoints intermédios a cada N épocas.",
    )
    parser.add_argument(
        "--critic_sigmoid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Se ativo (default), V(s)=sigmoid(linear); alinhar com alvos em [0,1].",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Cross-entropy com label smoothing (0=desligado; tentar 0.03–0.1).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (L2) no optimizador Adam.",
    )
    parser.add_argument(
        "--cosine_lr",
        action="store_true",
        help="Cosine annealing do LR até lr_min na última época.",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=0.0,
        help="LR mínimo com --cosine_lr (default 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (CUDA se existir, senão MPS no Mac, senão CPU), cpu, cuda, cuda:0, mps, …",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed precision (fp16) em CUDA; desliga com --no-amp se der NaN. Ignorado em CPU/MPS.",
    )
    parser.add_argument(
        "--rule_features",
        type=str,
        default="",
        help="Features extras de regras no obs: 'KEY:MIN:MAX,KEY2:MIN:MAX'. "
        "Ex.: STAB_MODIFIER:1.2:2.0,WEATHER_BOOST:1.0:2.0",
    )
    parser.add_argument(
        "--resample_rules_each_episode",
        action="store_true",
        help="Com --rule_features, reamostra valores das regras em cada reset.",
    )
    parser.add_argument(
        "--rule_conditioning",
        choices=("concat", "film", "film_per_layer"),
        default="concat",
        help="Como usar features de regra: concat (baseline), film, film_per_layer.",
    )
    parser.add_argument(
        "--film_hidden_dim",
        type=int,
        default=64,
        help="Dimensão oculta do MLP de FiLM.",
    )
    args = parser.parse_args()
    dataset_paths = [p.strip() for p in args.dataset if p.strip()]
    if not str(args.rule_features).strip() and dataset_paths:
        inferred = rule_features_cli_string_from_npz(dataset_paths[0])
        if inferred:
            args.rule_features = inferred
            print(f"Inferido --rule_features a partir de {dataset_paths[0]}: {args.rule_features}")
    if args.w_value > 0 and not dataset_paths:
        args.train_critic_online = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = pick_training_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler: torch_amp.GradScaler | None = (
        torch_amp.GradScaler("cuda") if use_amp else None
    )
    print(f"device={device}  amp={'on' if use_amp else 'off'}")

    hidden_dims = parse_hidden_dims(args.hidden_dims, args.hidden)

    rule_feature_keys, rule_feature_ranges = parse_rule_feature_ranges(args.rule_features)
    env = BattleEnv(
        encoder=args.encoder,
        n_active=args.n_active,
        max_team_size=args.max_team_size,
        max_pkm_moves=args.max_pkm_moves,
        rule_feature_keys=rule_feature_keys,
        rule_feature_ranges=rule_feature_ranges,
        resample_rules_on_reset=bool(args.resample_rules_each_episode),
    )
    expert = GreedyBattlePolicy()
    expert.params = env.params

    spec = BattleActionSpaceSpec.from_gym_space(env.action_space)
    model = BattleActorCritic(
        obs_dim=env.observation_space.shape[0],
        action_spec=spec,
        hidden_dims=hidden_dims,
        use_critic_sigmoid=bool(args.critic_sigmoid),
        rule_condition_dim=len(rule_feature_keys),
        rule_conditioning=args.rule_conditioning,
        film_hidden_dim=int(args.film_hidden_dim),
    )
    if args.init_ckpt.strip():
        ckpt = torch.load(args.init_ckpt.strip(), map_location="cpu")
        loaded, skipped, partial = load_compatible_weights(model, ckpt["model_state"])
        print(
            f"Inicializado com {args.init_ckpt} "
            f"(compatíveis: {loaded}, parciais: {partial}, ignorados: {skipped})"
        )
    model.to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=float(args.lr_min)
        )

    out_path = Path(args.out)
    stem = out_path.stem
    suffix = out_path.suffix if out_path.suffix else ".pt"

    ls = float(args.label_smoothing)
    if dataset_paths:
        obs_parts: list[np.ndarray] = []
        act_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        obs_dim_ref: int | None = None
        n_heads_ref: int | None = None
        for path in dataset_paths:
            data = np.load(path)
            o = data["obs"].astype(np.float32)
            a = data["actions"].astype(np.int64)
            v = data["critic_target"].astype(np.float32)
            if obs_dim_ref is None:
                obs_dim_ref = int(o.shape[1])
                n_heads_ref = int(a.shape[1])
            elif int(o.shape[1]) != obs_dim_ref or int(a.shape[1]) != n_heads_ref:
                raise ValueError(
                    f"Shapes incompatíveis ao juntar datasets: {path} obs={o.shape} actions={a.shape} "
                    f"vs primeiro ficheiro obs_dim={obs_dim_ref} n_heads={n_heads_ref}"
                )
            obs_parts.append(o)
            act_parts.append(a)
            val_parts.append(v)
        obs_np = np.concatenate(obs_parts, axis=0)
        act_np = np.concatenate(act_parts, axis=0)
        val_np = np.concatenate(val_parts, axis=0)
        print(
            f"Dataset: {len(dataset_paths)} ficheiros -> {obs_np.shape[0]} transições  "
            f"label_smoothing={ls}  weight_decay={args.weight_decay}"
        )
        n = obs_np.shape[0]
        bs = n if args.batch_size <= 0 else min(args.batch_size, n)
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = np.random.permutation(n)
            sum_tot = sum_lp = sum_lc = 0.0
            n_batches = 0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                obs = torch.from_numpy(obs_np[idx]).to(device)
                act = torch.from_numpy(act_np[idx]).to(device)
                opt.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
                    with torch_amp.autocast("cuda", dtype=torch.float16):
                        if args.w_value > 0:
                            val = torch.from_numpy(val_np[idx]).to(device)
                            total, l_p, l_c = model.supervised_actor_critic_loss(
                                obs, act, val, args.w_value, label_smoothing=ls
                            )
                        else:
                            total = model.supervised_loss(obs, act, label_smoothing=ls)
                            l_p, l_c = total.detach(), torch.tensor(0.0, device=device)
                    scaler.scale(total).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    if args.w_value > 0:
                        val = torch.from_numpy(val_np[idx]).to(device)
                        total, l_p, l_c = model.supervised_actor_critic_loss(
                            obs, act, val, args.w_value, label_smoothing=ls
                        )
                    else:
                        total = model.supervised_loss(obs, act, label_smoothing=ls)
                        l_p, l_c = total.detach(), torch.tensor(0.0, device=device)
                    total.backward()
                    opt.step()
                sum_tot += float(total.item())
                sum_lp += float(l_p.item())
                sum_lc += float(l_c.item())
                n_batches += 1
            mt = sum_tot / n_batches
            mp = sum_lp / n_batches
            mc = sum_lc / n_batches
            lr_s = opt.param_groups[0]["lr"]
            tail = f"  lr={lr_s:.2e}" if scheduler is not None else ""
            print(
                (
                    f"epoch {epoch:03d}  L={mt:.4f}  L_P={mp:.4f}  L_C={mc:.4f}{tail}"
                    if args.w_value > 0
                    else f"epoch {epoch:03d}  L_P={mp:.4f}{tail}"
                )
            )
            if scheduler is not None:
                scheduler.step()
            if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
                p = out_path.with_name(f"{stem}_ep{epoch:03d}{suffix}")
                save_checkpoint(p, model, spec, env, hidden_dims=hidden_dims, w_value=args.w_value)
                print(f"  checkpoint intermédio: {p.resolve()}")
    else:
        for epoch in range(1, args.epochs + 1):
            model.train()
            if args.w_value > 0 and args.train_critic_online:
                obs_np, act_np, val_np = collect_transitions_online(
                    env, expert, args.steps, with_values=True
                )
                obs = torch.from_numpy(obs_np).to(device)
                act = torch.from_numpy(act_np).to(device)
                val = torch.from_numpy(val_np).to(device)
            else:
                obs_np, act_np = collect_transitions_online(env, expert, args.steps, with_values=False)
                obs = torch.from_numpy(obs_np).to(device)
                act = torch.from_numpy(act_np).to(device)
                val = None
            opt.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch_amp.autocast("cuda", dtype=torch.float16):
                    if args.w_value > 0 and args.train_critic_online and val is not None:
                        total, l_p, l_c = model.supervised_actor_critic_loss(
                            obs, act, val, args.w_value, label_smoothing=ls
                        )
                    else:
                        total = model.supervised_loss(obs, act, label_smoothing=ls)
                        l_p, l_c = total.detach(), torch.tensor(0.0, device=device)
                scaler.scale(total).backward()
                scaler.step(opt)
                scaler.update()
            else:
                if args.w_value > 0 and args.train_critic_online and val is not None:
                    total, l_p, l_c = model.supervised_actor_critic_loss(
                        obs, act, val, args.w_value, label_smoothing=ls
                    )
                else:
                    total = model.supervised_loss(obs, act, label_smoothing=ls)
                    l_p, l_c = total.detach(), torch.tensor(0.0, device=device)
                total.backward()
                opt.step()
            lr_s = opt.param_groups[0]["lr"]
            tail = f"  lr={lr_s:.2e}" if scheduler is not None else ""
            print(
                (
                    f"epoch {epoch:03d}  L={total.item():.4f}  L_P={l_p.item():.4f}  L_C={l_c.item():.4f}{tail}"
                    if args.w_value > 0 and args.train_critic_online
                    else f"epoch {epoch:03d}  L_P={total.item():.4f}{tail}"
                )
            )
            if scheduler is not None:
                scheduler.step()
            if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
                p = out_path.with_name(f"{stem}_ep{epoch:03d}{suffix}")
                save_checkpoint(p, model, spec, env, hidden_dims=hidden_dims, w_value=args.w_value)
                print(f"  checkpoint intermédio: {p.resolve()}")

    save_checkpoint(out_path, model, spec, env, hidden_dims=hidden_dims, w_value=args.w_value)
    print(f"Guardado em {out_path.resolve()}")


if __name__ == "__main__":
    main()
