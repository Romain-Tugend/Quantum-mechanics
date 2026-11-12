#!/usr/bin/env python3
from pathlib import Path
import matplotlib

# Permet d'exécuter le script directement sur la front node du cluster sans écran
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent


def charger_csv(chemin_csv):
    data = np.genfromtxt(chemin_csv, delimiter=",", names=True)
    # Gestion du cas où une seule ligne est présente
    if data.ndim == 0:
        data = np.array([data])
    return data


def plot_strong(csv_path, out_png):
    data = charger_csv(csv_path)
    n = data["n"]
    mean = data["mean"]
    std = data["std"]
    speedup = data["speedup"]
    eff = data["efficiency"] * 100.0  # En pourcentage
    nx, ny = int(data["Nx"][0]), int(data["Ny"][0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    fig.suptitle(
        f"Scalabilité forte (Strong Scaling) — Grille {nx}×{ny}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Temps d'exécution
    axes[0].errorbar(
        n,
        mean,
        yerr=std,
        fmt="o-",
        color="#1f77b4",
        ecolor="#aec7e8",
        capsize=4,
        lw=2,
        label="Mesuré (moyenne ± std)",
    )
    axes[0].set_title("Temps d'exécution")
    axes[0].set_xlabel("Nombre de processus MPI (N)")
    axes[0].set_ylabel("Temps moyen (s)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 2. Accélération (Speedup)
    axes[1].plot(
        n, n, "--", color="gray", lw=1.5, label="Accélération idéale (S = N)"
    )
    axes[1].plot(
        n,
        speedup,
        "s-",
        color="#d62728",
        lw=2,
        label="Accélération mesurée T(1)/T(N)",
    )
    axes[1].set_title("Accélération (Speedup)")
    axes[1].set_xlabel("Nombre de processus MPI (N)")
    axes[1].set_ylabel("Speedup S(N)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # 3. Efficacité parallèle
    axes[2].axhline(
        100.0, linestyle="--", color="gray", lw=1.5, label="Idéal (100 %)"
    )
    axes[2].plot(
        n, eff, "^-", color="#2ca02c", lw=2, label="Efficacité parallèle S(N)/N"
    )
    axes[2].set_title("Efficacité parallèle")
    axes[2].set_xlabel("Nombre de processus MPI (N)")
    axes[2].set_ylabel("Efficacité (%)")
    axes[2].set_ylim(0, max(115, np.nanmax(eff) * 1.1))
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Graphique sauvegardé : {out_png}")


def plot_weak(csv_path, out_png):
    data = charger_csv(csv_path)
    n = data["n"]
    mean = data["mean"]
    std = data["std"]
    eff = data["efficiency"] * 100.0  # En pourcentage
    nx0, ny0 = int(data["Nx"][0]), int(data["Ny"][0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle(
        f"Scalabilité faible (Weak Scaling) — Charge de base {nx0}×{ny0} / CPU",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Temps d'exécution (idéalement constant)
    axes[0].axhline(
        mean[0],
        linestyle="--",
        color="gray",
        lw=1.5,
        label=f"Temps idéal T(1) = {mean[0]:.2f} s",
    )
    axes[0].errorbar(
        n,
        mean,
        yerr=std,
        fmt="o-",
        color="#1f77b4",
        ecolor="#aec7e8",
        capsize=4,
        lw=2,
        label="Mesuré (moyenne ± std)",
    )
    axes[0].set_title("Stabilité du temps d'exécution")
    axes[0].set_xlabel("Nombre de processus MPI (N)")
    axes[0].set_ylabel("Temps moyen (s)")
    axes[0].set_ylim(0, max(mean) * 1.25)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 2. Efficacité faible
    axes[1].axhline(
        100.0, linestyle="--", color="gray", lw=1.5, label="Idéal (100 %)"
    )
    axes[1].plot(
        n, eff, "^-", color="#9467bd", lw=2, label="Efficacité faible T(1)/T(N)"
    )
    axes[1].set_title("Efficacité en scalabilité faible")
    axes[1].set_xlabel("Nombre de processus MPI (N)")
    axes[1].set_ylabel("Efficacité (%)")
    axes[1].set_ylim(0, max(115, np.nanmax(eff) * 1.1))
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Graphique sauvegardé : {out_png}")


if __name__ == "__main__":
    strong_csv = REPO / "strong_results.csv"
    weak_csv = REPO / "weak_results.csv"

    if strong_csv.exists():
        plot_strong(strong_csv, REPO / "strong_scaling.png")
    else:
        print("Fichier strong_results.csv introuvable (ignoré).")

    if weak_csv.exists():
        plot_weak(weak_csv, REPO / "weak_scaling.png")
    else:
        print("Fichier weak_results.csv introuvable (ignoré).")
