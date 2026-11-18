#!/bin/bash
set -e

# 1. Compilation du projet
make -f makefile clean
make -f makefile

# 2. Exécution
mpirun ./a.out 0.0

for ((facteur = 1; facteur <= 7; facteur++)); do
    valeur_w=$(( 2**facteur ))
    mpirun ./a.out "${valeur_w}.0"
done

echo "Tests terminés. Résultats dans /data_files"
rm -f a.out
