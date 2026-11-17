import glob
import os
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Paramètres globaux ---
Nx = 256
Ny = 256
L = 20.0
A_X = 1.5        # Pas du réseau selon x (à aligner avec Potentiel.hpp)
A_Y = 2.0        # Pas du réseau selon y (à aligner avec Potentiel.hpp)
T_TOTAL = 30.0    # Temps physique total T (à aligner avec Parametres.hpp)

FPS_GIF = 20
MAX_FRAMES_GIF = 70

DOSSIER_DATA = "data_files"
DOSSIER_RESULTS = "results"
os.makedirs(DOSSIER_RESULTS, exist_ok=True)

lx = np.linspace(-L / 2, L / 2, Nx)
ly = np.linspace(-L / 2, L / 2, Ny)
extent = [-L / 2, L / 2, -L / 2, L / 2]
X, Y = np.meshgrid(lx, ly, indexing="ij")
points_par_frame = Nx * Ny

# --- 2. Recherche des simulations ---
motif = os.path.join(DOSSIER_DATA, "evolution_schrodinger2D_mpi_cas_E_x_*_Ey_*.bin")
fichiers_evo = sorted(glob.glob(motif))

if not fichiers_evo:
    print(f"Aucun fichier trouvé avec le motif : {motif}")
    exit(0)

print(f"{len(fichiers_evo)} simulation(s) détectée(s).\n")

# --- 3. Traitement de chaque cas ---
for chemin_evo in fichiers_evo:
    nom = os.path.basename(chemin_evo)
    match = re.search(r"E_x_([+-]?\d+\.\d+)_Ey_([+-]?\d+\.\d+)", nom)
    if not match:
        print(f"Nom non reconnu ({nom}) : ignoré.")
        continue

    str_Ex, str_Ey = match.group(1), match.group(2)
    Ex, Ey = float(str_Ex), float(str_Ey)
    label_E = f"Ex={Ex:g}, Ey={Ey:g}"

    chemin_pot = os.path.join(DOSSIER_DATA, f"potentiel_2D_cas_E_x_{str_Ex}_Ey_{str_Ey}.bin")
    nom_gif = os.path.join(DOSSIER_RESULTS, f"bloch_Ex_{Ex:g}_Ey_{Ey:g}.gif")
    nom_png = os.path.join(DOSSIER_RESULTS, f"analyse_barycentre_Ex_{Ex:g}_Ey_{Ey:g}.png")

    if not os.path.exists(chemin_pot):
        print(f"Potentiel manquant ({chemin_pot}) : ignoré.")
        continue

    psi2_all = np.fromfile(chemin_evo, dtype=np.float64)
    nb_frames = len(psi2_all) // points_par_frame
    if nb_frames == 0:
        continue

    V_all = np.fromfile(chemin_pot, dtype=np.float64)
    if V_all.size != points_par_frame:
        continue

    V = V_all.reshape((Nx, Ny)).T
    V_max, V_min = np.max(V), np.min(V)

    def get_matrix(frame):
        debut = frame * points_par_frame
        return psi2_all[debut : debut + points_par_frame].reshape((Nx, Ny)).T

    # Calcul des barycentres <x>(t) et <y>(t)
    x_bary = np.zeros(nb_frames)
    y_bary = np.zeros(nb_frames)
    for f in range(nb_frames):
        mat = get_matrix(f)
        norme = max(np.sum(mat), 1e-15)
        x_bary[f] = np.sum(mat * lx[None, :]) / norme
        y_bary[f] = np.sum(mat * ly[:, None]) / norme

    temps = np.linspace(0, T_TOTAL, nb_frames)
    initial_matrix = get_matrix(0)
    vmax = max(np.max(initial_matrix), 1e-12) / 2.0

    # =========================================================================
    # A. GÉNÉRATION DU GIF (Carte 2D + trajectoire du barycentre)
    # =========================================================================
    fig_gif, ax_gif = plt.subplots(figsize=(6.2, 5.2), dpi=80)
    im = ax_gif.imshow(initial_matrix, extent=extent, origin="lower", cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear")

    if V_max > V_min:
        niveaux = np.linspace(V_min, V_max, 6)[1:-1]
        ax_gif.contour(X, Y, V.T, levels=niveaux, colors="white", linewidths=0.7, alpha=0.25)

    trace_bary, = ax_gif.plot([], [], "-", color="cyan", lw=1.3, alpha=0.75)
    point_bary, = ax_gif.plot([x_bary[0]], [y_bary[0]], "+", color="cyan", markersize=10, markeredgewidth=2)
    ax_gif.set_xlabel("Position X")
    ax_gif.set_ylabel("Position Y")
    fig_gif.colorbar(im, ax=ax_gif, fraction=0.046, pad=0.04, label="|Ψ|²")

    def update_gif(frame):
        im.set_data(get_matrix(frame))
        debut = max(0, frame - 80)
        trace_bary.set_data(x_bary[debut : frame + 1], y_bary[debut : frame + 1])
        point_bary.set_data([x_bary[frame]], [y_bary[frame]])
        ax_gif.set_title(f"Bloch ({label_E}) — t = {temps[frame]:.2f}")
        return [im, trace_bary, point_bary]

    plt.tight_layout()
    indices_gif = np.linspace(0, nb_frames - 1, min(nb_frames, MAX_FRAMES_GIF), dtype=int)
    ani = animation.FuncAnimation(fig_gif, update_gif, frames=indices_gif, interval=1000 // FPS_GIF, blit=False)
    ani.save(nom_gif, writer=animation.PillowWriter(fps=FPS_GIF))
    plt.close(fig_gif)

    # =========================================================================
    # B. GÉNÉRATION DU PNG D'ANALYSE (4 panneaux : 2D, <x>(t), <y>(t), (<x>,<y>))
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), dpi=160)
    fig.suptitle(f"Analyse des oscillations de Bloch 2D ({label_E} | ax={A_X}, ay={A_Y})", fontsize=13, fontweight="bold")

    # 1. Densité finale + trajectoire complète
    ax0 = axes[0, 0]
    im0 = ax0.imshow(get_matrix(nb_frames - 1), extent=extent, origin="lower", cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear")
    if V_max > V_min:
        ax0.contour(X, Y, V.T, levels=niveaux, colors="white", linewidths=0.6, alpha=0.25)
    ax0.plot(x_bary, y_bary, "-", color="cyan", lw=1.2, alpha=0.85, label="Trajectoire ⟨r⟩(t)")
    ax0.set_title("Densité finale |Ψ|² et trajectoire")
    ax0.set_xlabel("X")
    ax0.set_ylabel("Y")
    ax0.legend(loc="upper left", facecolor="black", labelcolor="white", fontsize=8)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # 2. Courbe <x>(t)
    ax1 = axes[0, 1]
    ax1.plot(temps, x_bary, color="#1f77b4", lw=1.8)
    ax1.axhline(0, color="gray", linestyle=":", lw=1)
    if Ex != 0:
        TB_x = (2.0 * np.pi) / (abs(Ex) * A_X)
        for k in range(1, int(T_TOTAL // TB_x) + 1):
            ax1.axvline(k * TB_x, color="#d62728", linestyle="--", alpha=0.45, label="Période T_Bx" if k == 1 else "")
        ax1.set_title(f"Évolution de ⟨x⟩(t) — T_Bx théorique = {TB_x:.2f}")
        ax1.legend(fontsize=8)
    else:
        ax1.set_title("Évolution de ⟨x⟩(t) — Champ Ex = 0")
    ax1.set_xlabel("Temps t")
    ax1.set_ylabel("⟨x⟩(t)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 3. Courbe <y>(t)
    ax2 = axes[1, 0]
    ax2.plot(temps, y_bary, color="#2ca02c", lw=1.8)
    ax2.axhline(0, color="gray", linestyle=":", lw=1)
    if Ey != 0:
        TB_y = (2.0 * np.pi) / (abs(Ey) * A_Y)
        for k in range(1, int(T_TOTAL // TB_y) + 1):
            ax2.axvline(k * TB_y, color="#d62728", linestyle="--", alpha=0.45, label="Période T_By" if k == 1 else "")
        ax2.set_title(f"Évolution de ⟨y⟩(t) — T_By théorique = {TB_y:.2f}")
        ax2.legend(fontsize=8)
    else:
        ax2.set_title("Évolution de ⟨y⟩(t) — Champ Ey = 0")
    ax2.set_xlabel("Temps t")
    ax2.set_ylabel("⟨y⟩(t)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # 4. Trajectoire paramétrique (<x>(t), <y>(t)) colorée par le temps
    ax3 = axes[1, 1]
    sc = ax3.scatter(x_bary, y_bary, c=temps, cmap="viridis", s=12, zorder=3)
    ax3.plot(x_bary, y_bary, "-", color="gray", lw=0.8, alpha=0.5, zorder=2)
    ax3.plot(x_bary[0], y_bary[0], "ko", markersize=6, label="Départ (t=0)", zorder=4)
    ax3.set_title("Trajectoire paramétrique (⟨x⟩(t), ⟨y⟩(t))")
    ax3.set_xlabel("⟨x⟩")
    ax3.set_ylabel("⟨y⟩")
    ax3.axis("equal")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=8)
    fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="Temps t")

    plt.tight_layout()
    fig.savefig(nom_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {label_E} -> {nom_gif} et {nom_png}")
