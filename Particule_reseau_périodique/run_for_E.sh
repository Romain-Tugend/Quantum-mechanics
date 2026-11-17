#!/bin/bash
# 1. Compilation du projet
make -f makefile clean
make -f makefile
# 2. Execusion
# for VALEUR_E_x in 0.0 0.5 1.0 1.5 2.0 2.5 3.0; do
#     for VALEUR_E_y in 0.0 0.5 1.0 1.5 2.0 2.5 3.0; do
#         mpirun ./a.out $VALEUR_E_x $VALEUR_E_y
#     done
# done
mpirun ./a.out 0.4 0.0
mpirun ./a.out 0.4 0.4
mpirun ./a.out 0.45 0.3
mpirun ./a.out 0.6 0.2
echo "Tests terminés. Résultats dans /data_files"
rm a.out