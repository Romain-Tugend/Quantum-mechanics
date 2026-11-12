import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Paramètres ---
Nx = 256
Ny = 256
L = 20.0
X_ECRAN = (
    7.0  # Position de l'écran de détection (avant la zone absorbante à x=8.0)
)
NOM_IMAGE_PNG = "fentes_de_young_resultat.png"

FPS_INTERVAL = 40

print("Chargement des données...")
psi2_all = np.fromfile("evolution_schrodinger2D_mpi.bin", dtype=np.float64)

points_par_frame = Nx * Ny
nb_frames = len(psi2_all) // points_par_frame

if nb_frames == 0:
    raise ValueError("Aucune image complète trouvée dans le fichier.")

print(f"Détection de {nb_frames} images enregistrées.")

# Frame ciblée pour la sauvegarde PNG (à ~85 % de la simulation)
FRAME_CAPTURE = int(0.85 * (nb_frames - 1))

# Coordonnées spatiales
lx = np.linspace(-L / 2, L / 2, Nx)
ly = np.linspace(-L / 2, L / 2, Ny)
extent = [-L / 2, L / 2, -L / 2, L / 2]

X, Y = np.meshgrid(lx, ly, indexing="ij")


def get_matrix(frame):
    debut = frame * points_par_frame
    fin = debut + points_par_frame
    matrix = psi2_all[debut:fin].reshape((Nx, Ny))
    return matrix.T


# --- Extraction préalable des profils sur l'écran (x = X_ECRAN) ---
ix_ecran = np.argmin(np.abs(lx - X_ECRAN))
x_ecran_reel = lx[ix_ecran]

profils_instantanes = np.array(
    [get_matrix(f)[:, ix_ecran] for f in range(nb_frames)]
)
profils_cumules = np.cumsum(profils_instantanes, axis=0)

# Normalisation pour un affichage propre sur le panneau de droite
max_cumul = max(np.max(profils_cumules[-1]), 1e-12)
max_inst = max(np.max(profils_instantanes), 1e-12)

# --- Chargement du potentiel ---
print("Chargement du potentiel...")
V_all = np.fromfile("potentiel_2D.bin", dtype=np.float64)

if V_all.size != Nx * Ny:
    raise ValueError(
        f"Le potentiel contient {V_all.size} valeurs, mais {Nx*Ny} sont"
        " attendues."
    )

V = V_all.reshape((Nx, Ny)).T
V_max = np.max(V)
V_min = np.min(V)

# Calibration couleur (avec sécurité si t=0 est vide)
initial_matrix = get_matrix(0)
if np.max(initial_matrix) > 1e-12:
    vmax = np.max(initial_matrix) / 2.0
else:
    vmax = max(np.max(get_matrix(nb_frames // 2)), 1e-12) / 2.0

# --- Configuration du graphique (2 panneaux alignés en Y) ---
fig, (ax, ax_ecran) = plt.subplots(
    1,
    2,
    figsize=(12, 7.5),
    dpi=110,
    sharey=True,
    gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.08},
)

# 1. Panneau de gauche : Densité 2D |Psi|²
im = ax.imshow(
    initial_matrix,
    extent=extent,
    origin="lower",
    cmap="magma",
    vmin=0,
    vmax=vmax,
    interpolation="bilinear",
)

# Contours du mur et des fentes
if V_max > V_min:
    niveaux = np.linspace(V_min, V_max, 6)[1:-1]
    ax.contour(
        X,
        Y,
        V.T,
        levels=niveaux,
        colors="white",
        linewidths=0.8,
        alpha=0.35,
        linestyles="solid",
    )

# Ligne pointillée indiquant la position de l'écran sur la simulation 2D
ax.axvline(
    x_ecran_reel,
    color="cyan",
    linestyle="--",
    linewidth=1.2,
    alpha=0.7,
    label="Écran",
)
ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="black",
    labelcolor="white",
    fontsize=9,
)

ax.set_xlabel("Position X")
ax.set_ylabel("Position Y")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Densité de probabilité |Ψ|²")

# 2. Panneau de droite : Somme des probabilités sur l'écran
ax_ecran.set_facecolor("#140b24")
ax_ecran.set_xlim(0, 1.1)
ax_ecran.set_xlabel("Probabilité cumulée\n(Franges d'interférence)")
ax_ecran.set_title(f"Écran (x = {x_ecran_reel:.1f})", fontsize=10)
ax_ecran.grid(True, linestyle=":", alpha=0.25, color="white")

(line_cumul,) = ax_ecran.plot(
    profils_cumules[0] / max_cumul,
    ly,
    color="#feb72d",
    lw=2,
    label="Somme ∫|Ψ|² dt",
)
(line_inst,) = ax_ecran.plot(
    profils_instantanes[0] / max_inst,
    ly,
    color="white",
    lw=0.9,
    alpha=0.6,
    label="Instantané |Ψ|²",
)
fill_cumul = ax_ecran.fill_betweenx(
    ly, 0, profils_cumules[0] / max_cumul, color="#f1605d", alpha=0.45
)

ax_ecran.legend(
    loc="upper right",
    frameon=True,
    facecolor="black",
    labelcolor="white",
    fontsize=8,
)


# --- Animation ---
def update(frame):
    global fill_cumul
    matrix = get_matrix(frame)
    im.set_data(matrix)

    y_cumul = profils_cumules[frame] / max_cumul
    y_inst = profils_instantanes[frame] / max_inst

    line_cumul.set_xdata(y_cumul)
    line_inst.set_xdata(y_inst)

    fill_cumul.remove()
    fill_cumul = ax_ecran.fill_betweenx(
        ly, 0, y_cumul, color="#f1605d", alpha=0.45
    )

    ax.set_title(f"Fentes de Young — Image {frame}/{nb_frames - 1}")
    return [im, line_cumul, line_inst, fill_cumul]


# --- Sauvegarde immédiate de la capture PNG à 85 % de la simulation ---
update(FRAME_CAPTURE)
fig.savefig(NOM_IMAGE_PNG, dpi=200, bbox_inches="tight")
print(
    f"Image sauvegardée : {NOM_IMAGE_PNG} (frame"
    f" {FRAME_CAPTURE}/{nb_frames - 1})"
)

# --- Lancement de l'animation complète ---
ani = animation.FuncAnimation(
    fig,
    update,
    frames=nb_frames,
    interval=FPS_INTERVAL,
    blit=False,
    repeat=True,
)

plt.show()
