import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import PowerNorm
from matplotlib.colors import LogNorm


# --- 1. Paramètres (doivent correspondre au C++) ---
Nx = 512  
Ny = 512
L = 40.0

print("Chargement des données...")

# Extraction directe de la densité de probabilité |\Psi|^2
psi2_all = np.fromfile("evolution_schrodinger2D_mpi.bin", dtype=np.float64)

total_points = len(psi2_all)
points_par_frame = Nx * Ny
nb_frames = total_points // points_par_frame

print(f"Détection de {nb_frames} images enregistrées.")

# --- 2. Configuration de la figure Matplotlib ---
fig, ax = plt.subplots(figsize=(8, 6.5), dpi=120)

# Espacement pour éviter que le titre ne soit coupé
plt.subplots_adjust(top=0.86, left=0.15, right=0.95, bottom=0.12)

ax.set_xlabel("Axe X (Espace)", fontsize=11, labelpad=8)
ax.set_ylabel("Axe Y (Espace)", fontsize=11, labelpad=8)

# Première image linéaire brute
initial_matrix = psi2_all[0:points_par_frame].reshape((Nx, Ny)).T

# Calcul du contraste initial basé uniquement sur la zone APRÈS le mur (X > 0)
zone_droite_initiale = initial_matrix[:, Nx // 2:]
vmax_initial = np.max(zone_droite_initiale)

plage_decades = 4  # nombre d'ordres de grandeur affichés sous le pic
im = ax.imshow(initial_matrix, cmap="viridis", origin='lower',
               extent=[-L/2, L/2, -L/2, L/2],
               norm=LogNorm(vmin=(vmax_initial if vmax_initial > 0 else 1.0) / 10**plage_decades,
                             vmax=vmax_initial if vmax_initial > 0 else 1.0),
               interpolation='bilinear')

cbar = fig.colorbar(im, ax=ax, pad=0.04, shrink=0.85)
cbar.set_label("Densité de Probabilité $|\Psi|^2$ ", fontsize=10, labelpad=10)

# --- 3. Fonction de mise à jour pour l'animation ---
def update(frame):
    debut = frame * points_par_frame
    fin = debut + points_par_frame
    matrix = psi2_all[debut:fin].reshape((Nx, Ny)).T
    
    im.set_data(matrix)
    
    moitie_droite = matrix[:, Nx // 2:]
    vmax_courant_droite = np.max(moitie_droite)
    
    if vmax_courant_droite > 0:
        im.norm.vmax = vmax_courant_droite
        im.norm.vmin = vmax_courant_droite / 10**plage_decades
    
    ax.set_title(f"Évolution de $|\Psi|^2$ (Linéaire) — Image {frame}/{nb_frames}", fontsize=12, fontweight='bold', pad=15)
    
    return im,
# --- 4. Lancement de l'animation ---
ani = animation.FuncAnimation(
    fig, update, frames=nb_frames, interval=41, blit=False, repeat=True
)

plt.show()
