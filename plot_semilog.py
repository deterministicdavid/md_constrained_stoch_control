"""
Semi-log plot of value-function convergence against the predicted rate exp(-tau * eta * N).

Two panels (sup-norm error and relative L2 error), laid out like the convergence
figures written by run_convergence_experiment.py (figsize 10 x 3.8).

Usage (from the repo root, after the experiments have finished):
    python plot_semilog.py --floor 1e-11 --runs \
        convergence_results_disc_tau_0.1/convergence.csv:0.1 \
        convergence_results_disc_tau_0.05/convergence.csv:0.05 \
        convergence_results_disc_tau_0.02/convergence.csv:0.02

Options:
    --eta      mirror descent step size md_lambda (default 1.0)
    --floor    ignore errors below this level when fitting (numerical floor, default 1e-7)
    --algo     algo_label to plot (default "Mirror Descent")
    --title    figure title (default "[Disc] Value Function Convergence, varying tau")
    --figsize  width height in inches (default 10 3.8)
    --out      output file (default convergence_semilog.pdf)
"""
import argparse
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("value_linf", "o", r"Value $\|V_N - V_{ref}\|_\infty$"),
    ("value_rel_l2", "s", r"Relative Value $\|V_N - V_{ref}\|_2 / \|V_{ref}\|_2$"),
]


def load_run(path, algo):
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("algo_label") == algo]
    if not rows:
        sys.exit(f"No rows with algo_label '{algo}' in {path}")
    rows.sort(key=lambda r: float(r["n_steps"]))
    return rows


def tail_mask(Nv):
    tail = Nv >= 0.5 * Nv.max() if len(Nv) else np.array([], dtype=bool)
    if tail.sum() < 2:
        tail = np.zeros(len(Nv), dtype=bool)
        tail[-3:] = True
    return tail


def main():
    p = argparse.ArgumentParser(description="Semi-log convergence plot with predicted rates.")
    p.add_argument("--runs", nargs="+", required=True, help="entries of the form path/to/convergence.csv:tau")
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--floor", type=float, default=1e-7)
    p.add_argument("--algo", default="Mirror Descent")
    p.add_argument("--title", default=r"[Disc] Value Function Convergence, varying $\tau$")
    p.add_argument("--figsize", type=float, nargs=2, default=[10.0, 3.8])
    p.add_argument("--out", default="convergence_semilog.pdf")
    a = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=tuple(a.figsize))
    for run in a.runs:
        path, tau = run.rsplit(":", 1)
        tau = float(tau)
        rows = load_run(path, a.algo)
        N = np.array([float(r["n_steps"]) for r in rows])
        print(f"\ntau = {tau:g}  (predicted rate tau*eta = {tau * a.eta:.4f} per iteration)")

        for (key, marker, _), ax in zip(PANELS, axes):
            E = np.array([float(r[key]) for r in rows])
            (line,) = ax.semilogy(N, E, marker=marker, label=rf"$\tau={tau:g}$")

            valid = E > a.floor
            Nv, Ev = N[valid], E[valid]
            if key == "value_linf":
                for i in range(len(Nv) - 1):
                    r = (np.log(Ev[i]) - np.log(Ev[i + 1])) / (Nv[i + 1] - Nv[i])
                    print(f"  local rate N={Nv[i]:5g} -> {Nv[i + 1]:5g}: {r:.4f}")
                if len(Nv) < len(N):
                    print(f"  ({len(N) - len(Nv)} point(s) below floor {a.floor:g} not used)")

            tail = tail_mask(Nv)
            if tail.sum() >= 2:
                rate = -np.polyfit(Nv[tail], np.log(Ev[tail]), 1)[0]
                print(f"  [{key}] fitted rate on N in [{Nv[tail][0]:g}, {Nv[tail][-1]:g}]: {rate:.4f}")
                # predicted slope exp(-tau*eta*N), anchored at the last fitted point
                Ng = np.linspace(Nv[tail][0] * 0.5, Nv[tail][-1], 50)
                ax.semilogy(Ng, Ev[tail][-1] * np.exp(-tau * a.eta * (Ng - Nv[tail][-1])),
                            "--", color=line.get_color(), lw=1)

    for (_, _, title), ax in zip(PANELS, axes):
        ax.set_title(title)
        ax.set_xlabel("Iterations N")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title=r"dashed: $\propto e^{-\tau\eta N}$")
    fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()