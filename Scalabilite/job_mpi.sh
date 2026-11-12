#!/bin/bash -l
#SBATCH --job-name=SchrodingerMPI
#SBATCH --time=00:05:00
#SBATCH --output=output-%j.txt
#SBATCH --error=error-%j.txt

# Chargement des modules sur le nœud de calcul
module purge
module load openmpi
# module load fftw3  # Décommenter selon les modules du cluster

REPEATS=${1:-3}
NX=${2:-2048}
NY=${3:-2048}

# Exécution distribuée sur les nœuds attribués
for ((rep = 1; rep <= REPEATS; rep++)); do
  start=$(date +%s%N)
  mpirun ${SLURM_SUBMIT_DIR}/a.out "$NX" "$NY"
  status=$?
  end=$(date +%s%N)
  awk -v s="$start" -v e="$end" -v r="$rep" 'BEGIN {printf "wall_time_%d=%.6f\n", r, (e-s)/1e9}'
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done
