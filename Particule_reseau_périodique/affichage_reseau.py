import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Paramètres ---
Nx = 256
Ny = 256
L = 20.0
DOWNSAMPLE = 1 

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
extent = [-L/2, L/2, -L/2, L/2] 

X, Y = np.meshgrid(lx, ly, indexing="ij")

def get_matrix(frame):
    debut = frame * points_par_frame
    fin = debut + points_par_frame
    matrix = psi2_all[debut:fin].reshape((Nx, Ny))
    return matrix.T

# --- Chargement du potentiel ---
print("Chargement du potentiel...")
V_all = np.fromfile("potentiel_2D.bin", dtype=np.float64)

if V_all.size != Nx * Ny:
    raise ValueError(f"Le potentiel contient {V_all.size} valeurs, mais {Nx*Ny} sont attendues.")

V = V_all.reshape((Nx, Ny)).T
V_max = np.max(V)
V_min = np.min(V)

initial_matrix = get_matrix(0)
vmax = max(np.max(initial_matrix), 1e-12) / 2.0 

# --- Configuration du graphique 2D ---
fig, ax = plt.subplots(figsize=(10, 8), dpi=110)

# 1. Affichage de la densité |Psi|² (Heatmap)
im = ax.imshow(
    initial_matrix, 
    extent=extent, 
    origin='lower', 
    cmap='magma', 
    vmin=0, 
    vmax=vmax,
    interpolation='bilinear'
)

# 2. Superposition du potentiel (Réseau cristallin)
if V_max > V_min:
    # On crée 5 niveaux de contour pour bien voir la topologie des puits et des barrières
    niveaux = np.linspace(V_min, V_max, 6)[1:-1]
    ax.contour(
        X, Y, V.T, 
        levels=niveaux,
        colors='white', 
        linewidths=0.8, 
        alpha=0.3, # Semi-transparent pour ne pas gâcher la vue de l'onde
        linestyles='solid'
    )

ax.set_xlabel("Position X")
ax.set_ylabel("Position Y")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Densité de probabilité |Ψ|²")

# --- Animation ---
def update(frame):
    matrix = get_matrix(frame)
    im.set_data(matrix) 
    
    # Avec blit=False, on peut utiliser set_title classiquement
    ax.set_title(f"Dynamique quantique — Image {frame}/{nb_frames - 1}")
    
    return [im]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=nb_frames,
    interval=FPS_INTERVAL,
    blit=False, 
    repeat=True
)

plt.tight_layout()
plt.show()
