"""Carregar previsor de contagens e inferir a partir de X ou equipas."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from move_predictor_config import N_TARGETS, PKM0_SLICE, PKM1_SLICE, SWITCH_SLICE
from train_move_count_predictor import MLPGlobal, MLPLocalHeads, counts_to_move_dist, pick_device
from train_team_autoencoder import extract_team_matrix
from vgc2.util.encoding import EncodeContext, encode_team


def prepare_model_input(bundle: dict[str, Any], x: np.ndarray, *, rule_vec: np.ndarray | None = None) -> np.ndarray:
    """Converte X do dataset (equipas[+regras]) para o tensor que o .pt espera (996 ou latente+regras)."""
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    ae_ckpt = bundle.get("autoencoder_ckpt")
    if not ae_ckpt:
        return x_np

    from train_move_count_predictor_latent import encode_teams_with_ae  # noqa: PLC0415

    td = int(bundle.get("team_dim", 0))
    rd = int(bundle.get("rule_dim", 0))
    dev = bundle["device"]
    teams = extract_team_matrix(x_np, td) if td > 0 else x_np
    rv = rule_vec
    if rv is None and rd > 0 and x_np.shape[1] >= td + rd:
        rv = x_np[:, td : td + rd]
    return encode_teams_with_ae(teams, ae_ckpt, rule_vec=rv, device=dev)


def load_move_predictor(ckpt_path: str | Path, device: str | torch.device = "auto") -> dict[str, Any]:
    dev = pick_device(str(device)) if not isinstance(device, torch.device) else device
    ck = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    hidden = tuple(int(x) for x in ck["hidden_dims"])
    in_dim = int(ck["in_dim"])
    kind = str(ck["model_type"])
    if kind == "global":
        model = MLPGlobal(in_dim, hidden, N_TARGETS).to(dev)
    elif kind in ("local", "latent"):
        model = MLPLocalHeads(in_dim, hidden).to(dev)
    else:
        raise ValueError(f"model_type desconhecido: {kind}")
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return {
        "model": model,
        "device": dev,
        "model_type": kind,
        "in_dim": in_dim,
        "hidden_dims": hidden,
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "latent_dim": int(ck.get("latent_dim", 0)),
        "team_dim": int(ck.get("team_dim", 0)),
        "rule_dim": int(ck.get("rule_dim", 0)),
        "autoencoder_ckpt": ck.get("autoencoder_ckpt"),
    }


@torch.inference_mode()
def predict_move_counts(bundle: dict[str, Any], x: np.ndarray, *, rule_vec: np.ndarray | None = None) -> np.ndarray:
    """x: (in_dim,) ou (N, in_dim). Devolve contagens previstas (N, 9)."""
    model = bundle["model"]
    dev = bundle["device"]
    kind = bundle["model_type"]

    x_np = prepare_model_input(bundle, x, rule_vec=rule_vec)
    xb = torch.from_numpy(x_np).to(dev)
    if kind == "global":
        return model(xb).cpu().numpy().astype(np.float32)

    p0, p1, psw = model(xb)
    dist0 = F.softmax(p0, dim=-1)
    dist1 = F.softmax(p1, dim=-1)
    tot0 = torch.ones(xb.shape[0], 1, device=dev) * 3.0
    tot1 = torch.ones(xb.shape[0], 1, device=dev) * 3.0
    out = torch.cat([dist0 * tot0, dist1 * tot1, psw], dim=-1)
    return out.cpu().numpy().astype(np.float32)


def build_x_from_teams(
    team0,
    team1,
    *,
    rule_vec: np.ndarray | None = None,
    concat_rules: bool = True,
    ctx: EncodeContext | None = None,
) -> np.ndarray:
    ctx = ctx or EncodeContext()
    buf = np.zeros(20000, dtype=np.float32)
    a = encode_team(buf, team0, ctx)
    b = encode_team(buf[a:], team1, ctx)
    teams = buf[: a + b].copy()
    if concat_rules and rule_vec is not None and len(rule_vec) > 0:
        return np.concatenate([teams, np.asarray(rule_vec, dtype=np.float32)], axis=0)
    return teams


def action_profile_from_counts(counts9: np.ndarray) -> dict[str, float]:
    """Aproximação switch/damage/effect a partir de contagens agregadas (slots 0..7 + switch)."""
    c = np.asarray(counts9, dtype=np.float64).reshape(-1)
    if c.size != N_TARGETS:
        raise ValueError(f"Esperado {N_TARGETS} contagens, tem {c.size}")
    sw = float(max(c[SWITCH_SLICE], 0.0))
    moves = float(max(c[PKM0_SLICE].sum() + c[PKM1_SLICE].sum(), 0.0))
    total = sw + moves
    if total < 1e-9:
        return {"switch": 0.0, "damage": 0.0, "effect": 0.0}
    # Sem metadata por move: assume ~75% damage / 25% effect nos moves (heurística leve)
    freq_sw = sw / total
    freq_mv = moves / total
    return {"switch": freq_sw, "damage": 0.60 * freq_mv, "effect": 0.20 * freq_mv}


def balance_score_from_profile(profile: dict[str, float]) -> float:
    return (
        abs(0.20 - profile["switch"])
        + abs(0.60 - profile["damage"])
        + abs(0.20 - profile["effect"])
    )
