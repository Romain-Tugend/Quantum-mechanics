import matplotlib

matplotlib.use("Agg")  # Génère les fichiers sans ouvrir de fenêtre graphique
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Paramètres ---
Nx = 256
Ny = 256
L = 20.0
NOM_GIF = "reseau_cristallin.gif"
FPS_GIF = 20
MAX_FRAMES_GIF = 70  # Garde le GIF léger (~6-8 Mo, idéal pour GitHub)

print("Chargement des données...")
psi2_all = np.fromfile("evolution_schrodinger2D_mpi.bin", dtype=np.float64)

points_par_frame = Nx * Ny
nb_frames = len(psi2_all) // points_par_frame

if nb_frames == 0:
    raise ValueError("Aucune image complète trouvée dans le fichier.")

print(f"Détection de {nb_frames} images enregistrées.")

lx = np.linspace(-L / 2, L / 2, Nx)
ly = np.linspace(-L / 2, L / 2, Ny)
extent = [-L / 2, L / 2, -L / 2, L / 2]
X, Y = np.meshgrid(lx, ly, indexing="ij")


def get_matrix(frame):
    debut = frame * points_par_frame
    fin = debut + points_par_frame
    return psi2_all[debut:fin].reshape((Nx, Ny)).T


print("Chargement du potentiel...")
V_all = np.fromfile("potentiel_2D.bin", dtype=np.float64)
if V_all.size != Nx * Ny:
    raise ValueError(
        f"Le potentiel contient {V_all.size} valeurs, mais {Nx*Ny} sont"
        " attendues."
    )

V = V_all.reshape((Nx, Ny)).T
V_max, V_min = np.max(V), np.min(V)

initial_matrix = get_matrix(0)
vmax = max(np.max(initial_matrix), 1e-12) / 2.0

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


def update(frame):
    im.set_data(get_matrix(frame))
    ax.set_title(f"Dynamique quantique — Image {frame}/{nb_frames - 1}")
    return [im]


plt.tight_layout()

indices_frames = np.linspace(
    0, nb_frames - 1, min(nb_frames, MAX_FRAMES_GIF), dtype=int
)
ani = animation.FuncAnimation(
    fig, update, frames=indices_frames, interval=1000 // FPS_GIF, blit=False
)

print(f"Génération de {NOM_GIF} ({len(indices_frames)} frames)...")
ani.save(NOM_GIF, writer=animation.PillowWriter(fps=FPS_GIF))
plt.close(fig)
print(f"Terminé : {NOM_GIF}")
