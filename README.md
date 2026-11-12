<p align="center">
  <img src="Fentes%20de%20Young/fentes_de_young_resultat.png" alt="Résultat Fentes de Young" width="600">
</p>

Project of quantum mechanics during master 2 Ams at Paris-Saclay. 
It's the Young's interference experiment 1st version

We use the FFT from fftw librairy to solve the problem and the code is build for parallel computing architectures.

The Fentes de Young folder is the Young experiment, since the problem isn't periodique we have to add an other potential and solve the non homogen problem and we had a source that repeat in time to simulate a laser.

The Particule_reseau_périodique is the simulation of a crystal lattice in this case we have a good usage of the FFTW since cristalin lattice is periodic.

The Scalabilité folder is the crystal lattice simulation when we have remove the writting of the solution and it's juste pure calculation. We then proceed to run a weak and strong scalabily test