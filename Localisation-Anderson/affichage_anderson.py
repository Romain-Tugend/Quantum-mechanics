import glob
import os
import re
import matplotlib

matplotlib.use("Agg")  # Génère les fichiers sans ouvrir de fenêtre graphique
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Paramètres (alignés avec Parametres.hpp) ---
Nx = 256
Ny = 256
L = 20.0
T_TOTAL = 40.0  # T=20 dans Parametres.hpp
FPS_GIF = 20
MAX_FRAMES_GIF = 70  # Limite le poids du GIF (< 10 Mo pour GitHub)

DOSSIER_DATA = "data_files"
DOSSIER_RESULTS = "results"
os.makedirs(DOSSIER_RESULTS, exist_ok=True)

lx = np.linspace(-L / 2, L / 2, Nx)
ly = np.linspace(-L / 2, L / 2, Ny)
extent = [-L / 2, L / 2, -L / 2, L / 2]
X, Y = np.meshgrid(lx, ly, indexing="ij")
points_par_frame = Nx * Ny

# --- 2. Recherche automatique des simulations ---
motif = os.path.join(DOSSIER_DATA, "evolution_schrodinger2D_mpi_W_*.bin")
fichiers_evo = glob.glob(motif)

if not fichiers_evo:
    print(f"Aucun fichier trouvé dans '{DOSSIER_DATA}' avec le motif : {motif}")
    exit(0)

# Trie les fichiers par valeur numérique croissante de W
fichiers_evo.sort(key=lambda p: float(re.search(r"W_([+-]?\d+\.\d+)", p).group(1)))
print(f"{len(fichiers_evo)} simulation(s) Anderson détectée(s).\n")

courbes_variance = {}

# --- 3. Boucle sur chaque valeur de W ---
for chemin_evo in fichiers_evo:
    nom_fichier = os.path.basename(chemin_evo)
    str_W = re.search(r"W_([+-]?\d+\.\d+)", nom_fichier).group(1)
    W = float(str_W)
    label_W = f"{W:g}"

    chemin_pot = os.path.join(DOSSIER_DATA, f"potentiel_2D_W_{str_W}.bin")
    nom_gif = os.path.join(DOSSIER_RESULTS, f"anderson_W_{label_W}.gif")

    if not os.path.exists(chemin_pot):
        print(f"Potentiel manquant ({chemin_pot}) : simulation ignorée.")
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

    # Calcul de la variance spatiale sigma²(t) = <r²> - <r>²
    temps = np.linspace(0, T_TOTAL, nb_frames)
    variance_r2 = np.zeros(nb_frames)

    for f in range(nb_frames):
        mat = get_matrix(f)
        norme = max(np.sum(mat), 1e-15)
        xm = np.sum(mat * lx[None, :]) / norme
        ym = np.sum(mat * ly[:, None]) / norme
        x2m = np.sum(mat * (lx[None, :] ** 2)) / norme
        y2m = np.sum(mat * (ly[:, None] ** 2)) / norme
        variance_r2[f] = (x2m - xm**2) + (y2m - ym**2)

    courbes_variance[W] = (temps, variance_r2)

    # Génération du GIF pour ce niveau de désordre
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=80)
    initial_matrix = get_matrix(0)
    vmax = max(np.max(initial_matrix), 1e-12) * 0.5

    im = ax.imshow(
        initial_matrix,
        extent=extent,
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=vmax,
        interpolation="bilinear",
    )

    if W > 0 and V_max > V_min:
        niveaux = np.linspace(V_min, V_max, 5)[1:-1]
        ax.contour(X, Y, V.T, levels=niveaux, colors="white", linewidths=0.5, alpha=0.2)

    ax.set_xlabel("Position X")
    ax.set_ylabel("Position Y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Densité de probabilité |Ψ|²")

    def update(frame):
        im.set_data(get_matrix(frame))
        ax.set_title(f"Localisation d'Anderson (W = {label_W}) — t = {temps[frame]:.2f}")
        return [im]

    plt.tight_layout()
    indices_gif = np.linspace(0, nb_frames - 1, min(nb_frames, MAX_FRAMES_GIF), dtype=int)
    ani = animation.FuncAnimation(
        fig, update, frames=indices_gif, interval=1000 // FPS_GIF, blit=False
    )
    ani.save(nom_gif, writer=animation.PillowWriter(fps=FPS_GIF))
    plt.close(fig)
    print(f"[OK] GIF généré : {nom_gif}")

# --- 4. Graphique comparatif PNG : étalement sigma²(t) ---
fig_comp, ax_comp = plt.subplots(figsize=(9, 5.5), dpi=180)

for W, (t, var) in courbes_variance.items():
    if W == 0:
        ax_comp.plot(t, var, "--", color="black", lw=2.2, label="W = 0 (diffusion libre)")
    else:
        ax_comp.plot(t, var, "-", lw=2.0, label=f"W = {W:g}")

ax_comp.set_title("Transition vers la localisation d'Anderson en 2D", fontsize=13, fontweight="bold")
ax_comp.set_xlabel("Temps t")
ax_comp.set_ylabel("Étalement spatial σ²(t) = ⟨r²⟩ − ⟨r⟩²")
ax_comp.grid(True, linestyle="--", alpha=0.4)
ax_comp.legend()
plt.tight_layout()

chemin_png = os.path.join(DOSSIER_RESULTS, "comparaison_anderson_sigma2.png")
fig_comp.savefig(chemin_png, dpi=200, bbox_inches="tight")
plt.close(fig_comp)
print(f"\n[OK] Graphique comparatif sauvegardé : {chemin_png}")
