import glob
import os
import matplotlib

matplotlib.use("Agg")  # Génère les fichiers sans ouvrir de fenêtre graphique
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Paramètres globaux ---
Nx = 256
Ny = 256
L = 20.0
FPS_GIF = 20
MAX_FRAMES_GIF = 70  # Garde le GIF léger (~6-8 Mo, idéal pour GitHub)

DOSSIER_DATA = "data_files"
DOSSIER_RESULTS = "results"

# Crée le dossier results s'il n'existe pas encore
os.makedirs(DOSSIER_RESULTS, exist_ok=True)

# Grille spatiale (commune à toutes les simulations)
lx = np.linspace(-L / 2, L / 2, Nx)
ly = np.linspace(-L / 2, L / 2, Ny)
extent = [-L / 2, L / 2, -L / 2, L / 2]
X, Y = np.meshgrid(lx, ly, indexing="ij")
points_par_frame = Nx * Ny

# --- 2. Recherche automatique des fichiers d'évolution ---
motif_recherche = os.path.join(DOSSIER_DATA, "evolution_schrodinger2D_mpi_cas_E_*.bin")
fichiers_evolution = sorted(glob.glob(motif_recherche))

if not fichiers_evolution:
    print(f"Aucun fichier trouvé dans '{DOSSIER_DATA}' avec le motif '{motif_recherche}'.")
    exit(0)

print(f"{len(fichiers_evolution)} simulation(s) détectée(s).\n")

# --- 3. Boucle sur chaque simulation ---
for chemin_evo in fichiers_evolution:
    nom_fichier = os.path.basename(chemin_evo)
    # Extrait la partie dynamique (ex: "0.500000") depuis le nom du fichier
    suffixe_E = nom_fichier.replace("evolution_schrodinger2D_mpi_cas_E_", "").replace(".bin", "")
    
    chemin_pot = os.path.join(DOSSIER_DATA, f"potentiel_2D_cas_E_{suffixe_E}.bin")
    nom_gif = os.path.join(DOSSIER_RESULTS, f"oscillations_bloch_E_{suffixe_E}.gif")

    print(f"--- Traitement de E = {suffixe_E} ---")

    # Vérifie que le fichier de potentiel correspondant existe bien
    if not os.path.exists(chemin_pot):
        print(f" Potentiel manquant ({chemin_pot}) : simulation ignorée.\n")
        continue

    # Chargement de l'évolution
    psi2_all = np.fromfile(chemin_evo, dtype=np.float64)
    nb_frames = len(psi2_all) // points_par_frame

    if nb_frames == 0:
        print(f" Fichier vide ou incomplet ({chemin_evo}) : simulation ignorée.\n")
        continue

    print(f"Détection de {nb_frames} images enregistrées.")

    # Chargement du potentiel associé
    V_all = np.fromfile(chemin_pot, dtype=np.float64)
    if V_all.size != points_par_frame:
        print(f" Taille du potentiel incorrecte ({V_all.size} au lieu de {points_par_frame}) : ignorée.\n")
        continue

    V = V_all.reshape((Nx, Ny)).T
    V_max, V_min = np.max(V), np.min(V)

    def get_matrix(frame):
        debut = frame * points_par_frame
        fin = debut + points_par_frame
        return psi2_all[debut:fin].reshape((Nx, Ny)).T

    initial_matrix = get_matrix(0)
    vmax = max(np.max(initial_matrix), 1e-12) / 2.0

    # Création de la figure
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=80)

    im = ax.imshow(
        initial_matrix,
        extent=extent,
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=vmax,
        interpolation="bilinear",
    )

    if V_max > V_min:
        niveaux = np.linspace(V_min, V_max, 6)[1:-1]
        ax.contour(
            X,
            Y,
            V.T,
            levels=niveaux,
            colors="white",
            linewidths=0.8,
            alpha=0.3,
            linestyles="solid",
        )

    ax.set_xlabel("Position X")
    ax.set_ylabel("Position Y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Densité de probabilité |Ψ|²")

    # Formate la valeur de E pour un affichage propre dans le titre (ex: 0.500000 -> 0.5)
    valeur_E_affichee = f"{float(suffixe_E):g}"

    def update(frame):
        im.set_data(get_matrix(frame))
        ax.set_title(f"Oscillations de Bloch (E = {valeur_E_affichee}) — Image {frame}/{nb_frames - 1}")
        return [im]

    plt.tight_layout()

    indices_frames = np.linspace(
        0, nb_frames - 1, min(nb_frames, MAX_FRAMES_GIF), dtype=int
    )
    ani = animation.FuncAnimation(
        fig, update, frames=indices_frames, interval=1000 // FPS_GIF, blit=False
    )

    print(f"Génération de {nom_gif} ({len(indices_frames)} frames)...")
    ani.save(nom_gif, writer=animation.PillowWriter(fps=FPS_GIF))
    plt.close(fig)
    print(f"Terminé : {nom_gif}\n")
