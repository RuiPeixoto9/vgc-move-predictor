"""Validação do previsor com KL divergence + gráficos (reunião 2026-05-22).

Métrica (Wikipedia / orientador, 2026-05-22):
  KL(P || Q)  com  **P = target Y (true)**  e  **Q = rede (approximation)**
  i.e. KL(true || approx) = KL(Y || rede)  — **não é simétrica**.

Distribuições: vetor de 9 dimensões (2×4 moves + switch), soma = 1.
  - P_i = Y_i normalizado (true)
  - Q_i = saída da rede normalizada (approx)

Gráficos: histogramas sobrepostos (Y vs rede) para amostras do conjunto de validação.

Uso:
  python -u eval_move_predictor_kl.py \\
    --ckpt move_predictor_local.pt \\
    --data move_count_sup_switch.npz \\
    --out_dir plots_kl_local
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from move_predictor_config import N_TARGETS, PKM0_SLICE, PKM1_SLICE, SWITCH_SLICE
from move_predictor_io import load_move_predictor

SLOT_LABELS = (
    "P0-M0",
    "P0-M1",
    "P0-M2",
    "P0-M3",
    "P1-M0",
    "P1-M1",
    "P1-M2",
    "P1-M3",
    "SW",
)


def counts_to_prob(y: np.ndarray, *, eps: float = 0.01) -> np.ndarray:
    """Contagens -> prob. Laplace (y+eps)/sum(y+K*eps), para KL estável com slots a zero."""
    a = np.asarray(y, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    a = np.clip(a, 0.0, None) + float(eps)
    return a / a.sum(axis=-1, keepdims=True)


@torch.inference_mode()
def predict_prob_9(bundle: dict, x: np.ndarray, *, eps: float = 0.01) -> np.ndarray:
    """Massas não negativas (N,9) antes de normalizar com o mesmo eps que Y."""
    import torch.nn.functional as F

    from move_predictor_io import predict_move_counts

    raw = predict_move_counts(bundle, x)
    raw = np.clip(raw, 0.0, None) + float(eps)

    # Local: reforçar shape das cabeças (softmax nos 8 moves)
    if bundle["model_type"] in ("local", "latent"):
        from move_predictor_io import prepare_model_input

        model = bundle["model"]
        dev = bundle["device"]
        x_np = prepare_model_input(bundle, x)
        xb = torch.from_numpy(x_np).to(dev)
        p0, p1, psw = model(xb)
        d0 = F.softmax(p0, dim=-1).cpu().numpy()
        d1 = F.softmax(p1, dim=-1).cpu().numpy()
        sw = np.clip(psw.cpu().numpy(), 0.0, None) + eps
        tot_move = np.maximum(raw[:, PKM0_SLICE].sum(1) + raw[:, PKM1_SLICE].sum(1), eps)
        for i in range(raw.shape[0]):
            raw[i, PKM0_SLICE] = d0[i] * tot_move[i] * 0.5
            raw[i, PKM1_SLICE] = d1[i] * tot_move[i] * 0.5
            raw[i, SWITCH_SLICE] = sw[i]
    return raw / raw.sum(axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """KL(P || Q) por amostra (ln). p=primeiro arg (true), q=segundo (approx)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def _val_split(n: int, val_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    return perm[:n_val]


def _plot_overlay(
    q: np.ndarray,
    p: np.ndarray,
    path: Path,
    *,
    title: str,
    kl_pq: float,
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(N_TARGETS)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, q, width=width, label="target Y (normalizado)", color="#4c72b0", alpha=0.75)
    ax.bar(x + width / 2, p, width=width, label="rede (normalizado)", color="#dd8452", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(SLOT_LABELS, rotation=45, ha="right")
    ax.set_ylabel("probabilidade")
    ax.set_ylim(0, max(0.05, float(max(q.max(), p.max()) * 1.15)))
    ax.set_title(f"{title}\nKL(true || rede) = {kl_pq:.4f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_overlay_lines(
    q: np.ndarray,
    p: np.ndarray,
    path: Path,
    *,
    title: str,
    kl_pq: float,
) -> None:
    """Curvas sobrepostas (pedido do orientador)."""
    import matplotlib.pyplot as plt

    x = np.arange(N_TARGETS)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, q, alpha=0.35, color="#4c72b0", label="target Y")
    ax.plot(x, q, "o-", color="#4c72b0", linewidth=2, markersize=6)
    ax.fill_between(x, p, alpha=0.35, color="#dd8452", label="rede")
    ax.plot(x, p, "s-", color="#dd8452", linewidth=2, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(SLOT_LABELS, rotation=45, ha="right")
    ax.set_ylabel("probabilidade")
    ax.set_title(f"{title}\nKL(true || rede) = {kl_pq:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="KL(rede||Y) + plots de distribuição de uso de moves.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_dir", type=str, default="plots_kl")
    ap.add_argument("--n_plot", type=int, default=5, help="Nº de exemplos a desenhar (do val set).")
    ap.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Laplace smoothing nas distribuições (evita KL infinita em slots com count=0).",
    )
    ap.add_argument(
        "--also_kl_reverse",
        action="store_true",
        help="Também imprime KL(rede||Y) (ordem antiga, inversa à Wikipedia).",
    )
    ap.add_argument("--skip_plots", action="store_true", help="Só estatísticas KL (sem matplotlib).")
    ap.add_argument(
        "--best_worst_only",
        action="store_true",
        help="Só 4 PNGs: best_bars/curves + worst_bars/curves (melhor e pior KL).",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Apaga bars_*.png e curves_*.png antigos na pasta de saída antes de plotar.",
    )
    ap.add_argument(
        "--kl_histogram",
        action="store_true",
        help="Histograma da distribuição de KL(true||rede) no val set.",
    )
    ap.add_argument(
        "--median_plot",
        action="store_true",
        help="Gráfico do exemplo com KL mediana (além de best/worst se --best_worst_only).",
    )
    args = ap.parse_args()
    if args.best_worst_only:
        args.n_plot = 3 if args.median_plot else 2

    d = np.load(args.data, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=np.float32)
    n = X.shape[0]
    val_idx = _val_split(n, args.val_fraction, args.seed)

    bundle = load_move_predictor(args.ckpt, args.device)
    P_true = counts_to_prob(Y[val_idx], eps=args.eps)
    Q_approx = predict_prob_9(bundle, X[val_idx], eps=args.eps)

    # KL(true || approx) — Wikipedia: P=true, Q=approximation
    kl_main = kl_divergence(P_true, Q_approx, eps=args.eps)
    kl_main_report = np.maximum(kl_main, 0.0)

    print(f"ckpt={Path(args.ckpt).name}  model={bundle['model_type']}  val_n={len(val_idx)}")
    print(
        "KL(true || rede)  [P=Y, Q=rede]  "
        f"mean={kl_main_report.mean():.4f}  median={np.median(kl_main_report):.4f}  "
        f"std={kl_main_report.std():.4f}"
    )
    print(
        f"                  min={kl_main_report.min():.4f}  max={kl_main_report.max():.4f}  "
        f"(bruto min={kl_main.min():.4f})"
    )

    if args.also_kl_reverse:
        kl_alt = kl_divergence(Q_approx, P_true, eps=args.eps)
        print(f"KL(rede || Y) mean={np.maximum(kl_alt, 0.0).mean():.4f}  (ordem inversa)")

    for name, sl in ("pkm0", PKM0_SLICE), ("pkm1", PKM1_SLICE), ("switch", SWITCH_SLICE):
        pt = counts_to_prob(Y[val_idx, sl], eps=args.eps)
        qa = counts_to_prob(Q_approx[:, sl], eps=args.eps)
        kg = kl_divergence(pt, qa)
        print(f"  KL(true||rede) [{name}] mean={np.maximum(kg, 0.0).mean():.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(kl_main_report)

    # Guardar estatísticas
    stats_path = out_dir / "kl_stats.npz"
    best_i = int(order[0]) if len(order) else -1
    worst_i = int(order[-1]) if len(order) else -1
    np.savez_compressed(
        stats_path,
        kl_true_approx=kl_main.astype(np.float32),
        kl_true_approx_clipped=kl_main_report.astype(np.float32),
        kl_order="true||approx (P=Y, Q=rede)",
        val_idx=val_idx,
        mean_kl=float(kl_main_report.mean()),
        best_kl=float(kl_main_report[best_i]) if best_i >= 0 else np.nan,
        worst_kl=float(kl_main_report[worst_i]) if worst_i >= 0 else np.nan,
        best_dataset_idx=int(val_idx[best_i]) if best_i >= 0 else -1,
        worst_dataset_idx=int(val_idx[worst_i]) if worst_i >= 0 else -1,
        ckpt=str(Path(args.ckpt).resolve()),
    )
    print(f"stats: {stats_path.resolve()}")

    if args.kl_histogram and not args.skip_plots:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(kl_main_report, bins=50, color="#4c72b0", alpha=0.85, edgecolor="white")
            ax.axvline(float(kl_main_report.mean()), color="#c44e52", linestyle="--", linewidth=2, label=f"média={kl_main_report.mean():.3f}")
            ax.axvline(float(np.median(kl_main_report)), color="#55a868", linestyle=":", linewidth=2, label=f"mediana={np.median(kl_main_report):.3f}")
            ax.set_xlabel("KL(true || rede)")
            ax.set_ylabel("nº amostras")
            ax.set_title(f"Distribuição KL — val set (n={len(val_idx)})")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            hist_path = out_dir / "kl_histogram.png"
            fig.savefig(hist_path, dpi=120)
            plt.close(fig)
            print(f"  histograma KL -> {hist_path.name}")
        except ImportError:
            print("AVISO: --kl_histogram requer matplotlib.")

    if args.skip_plots:
        if not args.kl_histogram:
            print("plots: ignorados (--skip_plots)")
        return

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "AVISO: matplotlib não instalado — KL guardada em kl_stats.npz, sem PNG.\n"
            "  pip install matplotlib\n"
            "  Depois volta a correr este script (sem --skip_plots)."
        )
        return

    if args.clean:
        for pat in ("bars_*.png", "curves_*.png", "best_*.png", "worst_*.png"):
            for old in out_dir.glob(pat):
                old.unlink(missing_ok=True)

    if len(order):
        print(
            f"  melhor KL(true||rede)={float(kl_main_report[order[0]]):.4f} "
            f"(dataset idx {int(val_idx[order[0]])})  "
            f"pior KL={float(kl_main_report[order[-1]]):.4f} (dataset idx {int(val_idx[order[-1]])})"
        )

    if args.best_worst_only:
        picks: list[tuple[str, int]] = []
        if len(order) > 0:
            picks.append(("best", int(order[0])))
        if len(order) > 1:
            picks.append(("worst", int(order[-1])))
        if args.median_plot and len(order) > 0:
            picks.append(("median", int(order[len(order) // 2])))
    else:
        picks = []
        if len(order) > 0:
            picks.append(("sample", int(order[0])))
        if len(order) > 1:
            picks.append(("sample", int(order[-1])))
        rng = np.random.default_rng(args.seed + 1)
        used = {p[1] for p in picks}
        rest = [i for i in range(len(val_idx)) if i not in used]
        extra = max(0, args.n_plot - len(picks))
        if rest and extra:
            for i in rng.choice(rest, size=min(extra, len(rest)), replace=False):
                picks.append(("sample", int(i)))

    for j, (label, i) in enumerate(picks[: args.n_plot]):
        qi = P_true[i]
        pi = Q_approx[i]
        k = float(kl_main_report[i])
        ds_idx = int(val_idx[i])
        if args.best_worst_only or label in ("best", "worst", "median"):
            prefix = label  # best | worst | median
            title = f"{prefix.upper()} — val_idx={ds_idx}  KL(true||rede)={k:.4f}"
            _plot_overlay(qi, pi, out_dir / f"{prefix}_bars.png", title=title, kl_pq=k)
            _plot_overlay_lines(qi, pi, out_dir / f"{prefix}_curves.png", title=title, kl_pq=k)
            print(f"  {prefix}: KL(true||rede)={k:.4f}  -> {prefix}_bars.png, {prefix}_curves.png")
        else:
            tag = f"val_{ds_idx}_kl{k:.4f}"
            title = f"val_idx={ds_idx}  KL(true||rede)={k:.4f}"
            _plot_overlay(qi, pi, out_dir / f"bars_{j:02d}_{tag}.png", title=title, kl_pq=k)
            _plot_overlay_lines(qi, pi, out_dir / f"curves_{j:02d}_{tag}.png", title=title, kl_pq=k)

    print(f"plots: {out_dir.resolve()}  ({len(picks[: args.n_plot])} exemplos)")


if __name__ == "__main__":
    main()
