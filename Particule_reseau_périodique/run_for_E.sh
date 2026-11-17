#!/bin/bash
# 1. Compilation du projet
make -f makefile clean
make -f makefile
# 2. Execusion
for VALEUR_E in 0.0 0.5 1.0 2.0; do
    mpirun ./a.out $VALEUR_E
done
echo "Tests terminés. Résultats dans /data_files"
rm a.out