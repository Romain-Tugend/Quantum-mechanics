# 2D Time-Dependent Schrödinger Equation Solver (MPI / FFTW)

<p align="center">
  <img src="Fentes%20de%20Young/results/fentes_de_young.gif" alt="Young's Double-Slit Simulation" width="650">
</p>

Project of quantum mechanics during **Master 2 Ams at Paris-Saclay**. 
We use the FFT from fftw librairy to solve the problem and the code is build for parallel computing architectures.

## Repository Structure

### 1. `Fentes de Young` (Young's Double-Slit Experiment)

The Fentes de Young folder is the Young experiment, since the problem isn't periodique we have to add an other potential and solve the non homogen problem additionally we've added a source that repeat in time to simulate a laser.

<p align="center">
  <img src="Fentes%20de%20Young/results/fentes_de_young_resultat.png" alt="Young's Interference Pattern" width="650">
</p>

### 2. `Particule_reseau_périodique` (Particle in a Crystal Lattice)
The Particule_reseau_périodique is the simulation of a crystal lattice in this case we have a good usage of the FFTW since cristalin lattice is periodic.

<p align="center">
  <img src="Particule_reseau_p%C3%A9riodique/reseau_cristallin.gif" alt="2D Schrödinger Simulation in a Crystal Lattice" width="550">
</p>

We see the Bloch oscillations:

<p align="center">
  <img src="Particule_reseau_p%C3%A9riodique/results_Bloch/bloch_Ex_0.45_Ey_0.3.gif" alt="2D Schrödinger Simulation in a Crystal Lattice" width="550">
</p>

We can take a look at the expectation value for $`x`$: 
$$
\langle x \rangle_{\psi}(t) = \iint x \lvert \psi(x, y, t) \rvert^2 \, dx \, dy
$$
Doing the same for $`y`$ we can plot the trajectory:
$$
\bigl(\langle x \rangle_{\psi}(t), \, \langle y \rangle_{\psi}(t)\bigr)
$$
We see that the curve is circle-like.

<p align="center">
  <img src="Particule_reseau_p%C3%A9riodique/results_Bloch/bloch_Ex_0.45_Ey_0.3.gif" alt="Young's Interference Pattern" width="650">
</p>

### 3. `Scalabilite` (HPC Benchmarks)

The Scalabilité folder is the crystal lattice simulation where we have removed the writting part of the solution and it's juste pure calculation. We then proceed to run a weak and strong scalabily test.

<p align="center">
  <img src="Scalabilite/weak_scaling.png" alt="Results weak scalabily" width="600">
</p>
<p align="center">
  <img src="Scalabilite/strong_scaling.png" alt="Results strong scalabily" width="600">
</p>