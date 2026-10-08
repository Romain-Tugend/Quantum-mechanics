import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour projection='3d'

# --- 1. Paramètres (doivent correspondre au C++) ---
Nx = 512
Ny = 512
L = 40.0

# Pas de sous-échantillonnage pour l'affichage 3D (512x512 est trop lourd à tracer)
DOWNSAMPLE = 4  # augmente (8, 16...) si c'est encore trop lent

print("Chargement des données...")
psi2_all = np.fromfile("evolution_schrodinger2D_mpi.bin", dtype=np.float64)

total_points = len(psi2_all)
points_par_frame = Nx * Ny
nb_frames = total_points // points_par_frame
print(f"Détection de {nb_frames} images enregistrées.")

# Grille physique (x, y) pour le plot 3D
lx = np.linspace(-L/2, L/2, Nx)
ly = np.linspace(-L/2, L/2, Ny)
X, Y = np.meshgrid(lx[::DOWNSAMPLE], ly[::DOWNSAMPLE], indexing='ij')

def get_matrix(frame):
    debut = frame * points_par_frame
    fin = debut + points_par_frame
    matrix = psi2_all[debut:fin].reshape((Nx, Ny))
    return matrix[::DOWNSAMPLE, ::DOWNSAMPLE]

# --- 2. Configuration de la figure 3D ---
fig = plt.figure(figsize=(9, 7), dpi=120)
ax = fig.add_subplot(111, projection='3d')

initial_matrix = get_matrix(0)
vmax_initial = np.max(initial_matrix)

# Vue "de côté" : on regarde quasiment dans le plan (X, Z), depuis le côté
ax.view_init(elev=25, azim=-45)

ax.set_xlabel("Axe X (Espace)")
ax.set_ylabel("Axe Y (Espace)")
ax.set_zlabel("Densité $|\Psi|^2$")
ax.set_zlim(0, vmax_initial if vmax_initial > 0 else 1.0)

surf = [ax.plot_surface(X, Y, initial_matrix, cmap="viridis",
                         vmin=0, vmax=vmax_initial, linewidth=0, antialiased=False)]

# --- 3. Fonction de mise à jour ---
def update(frame):
    matrix = get_matrix(frame)

    surf[0].remove()
    surf[0] = ax.plot_surface(X, Y, matrix, cmap="viridis",
                               vmin=0, vmax=vmax_initial,
                               linewidth=0, antialiased=False)
    ax.set_title(f"Évolution de $|\Psi|^2$ — Image {frame}/{nb_frames}", fontsize=12, fontweight='bold')

    return surf[0],

# --- 4. Lancement de l'animation ---
ani = animation.FuncAnimation(
    fig, update, frames=nb_frames, interval=41, blit=False, repeat=True
)

plt.show()