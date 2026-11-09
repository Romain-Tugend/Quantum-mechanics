import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Paramètres ---
Nx = 256
Ny = 256
L = 20.0
DOWNSAMPLE = 1 # Mis à 1 pour une résolution maximale (la 2D est rapide)

FPS_INTERVAL = 40

print("Chargement des données...")
psi2_all = np.fromfile(
    "evolution_schrodinger2D_mpi.bin",
    dtype=np.float64
)

points_par_frame = Nx * Ny
nb_frames = len(psi2_all) // points_par_frame

if nb_frames == 0:
    raise ValueError("Aucune image complète trouvée dans le fichier.")

print(f"Détection de {nb_frames} images enregistrées.")

# Coordonnées spatiales
lx = np.linspace(-L / 2, L / 2, Nx)
ly = np.linspace(-L / 2, L / 2, Ny)
extent = [-L/2, L/2, -L/2, L/2] # Pour aligner l'image sur les axes physiques

X, Y = np.meshgrid(
    lx[::DOWNSAMPLE],
    ly[::DOWNSAMPLE],
    indexing="ij"
)

def get_matrix(frame):
    debut = frame * points_par_frame
    fin = debut + points_par_frame
    # On lit et on transpose (.T) pour que l'axe X (lignes en C++) 
    # corresponde bien à l'axe horizontal sur l'image matplotlib
    matrix = psi2_all[debut:fin].reshape((Nx, Ny))
    return matrix[::DOWNSAMPLE, ::DOWNSAMPLE].T

# --- Chargement du potentiel ---
print("Chargement du potentiel...")
V_all = np.fromfile("potentiel_2D.bin", dtype=np.float64)

if V_all.size != Nx * Ny:
    raise ValueError(
        f"Le potentiel contient {V_all.size} valeurs, "
        f"mais {Nx} x {Ny} = {Nx * Ny} sont attendues."
    )

V = V_all.reshape((Nx, Ny))[::DOWNSAMPLE, ::DOWNSAMPLE].T
V_max = np.max(V)

initial_matrix = get_matrix(0)
# On fixe une échelle max (vmax) pour éviter que les couleurs clignotent
# On peut diviser par un facteur (ex: 2 ou 5) pour saturer un peu et mieux voir les franges faibles
vmax = max(np.max(initial_matrix), 1e-12) / 2.0 

# --- Configuration du graphique 2D ---
fig, ax = plt.subplots(figsize=(10, 8), dpi=110)

# 1. Affichage de la densité |Psi|² (Heatmap)
# 'magma' ou 'inferno' sont excellents pour faire ressortir les ondes
im = ax.imshow(
    initial_matrix, 
    extent=extent, 
    origin='lower', 
    cmap='magma', 
    vmin=0, 
    vmax=vmax,
    interpolation='bilinear'
)

# 2. Superposition du potentiel (Mur et fentes) en contour blanc
if V_max > 0:
    ax.contour(
        X, Y, V.T, # V.T car contour utilise X, Y directements
        levels=[V_max * 0.1], # Ligne de niveau à 10% de la hauteur du mur
        colors='white', 
        linewidths=1.5, 
        alpha=0.7,
        linestyles='dashed'
    )

ax.set_xlabel("Position X")
ax.set_ylabel("Position Y")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Densité de probabilité |Ψ|²")

# --- Animation ---
def update(frame):
    matrix = get_matrix(frame)
    im.set_data(matrix) # Mise à jour ultra-rapide des pixels
    ax.set_title(f"Interférences Quantiques — Image {frame}/{nb_frames - 1}")
    return [im]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=nb_frames,
    interval=FPS_INTERVAL,
    blit=True, # Blit=True rend l'animation beaucoup plus fluide en 2D
    repeat=True
)

plt.tight_layout()
plt.show()
