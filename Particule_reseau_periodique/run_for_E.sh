#!/bin/bash
# 1. Compilation du projet
make -f makefile clean
make -f makefile
# 2. Execusion
# for VALEUR_E_y in 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0; do
#     VALEUR_E_x=$(echo "$VALEUR_E_y / 1.5" | bc -l)
#     mpirun ./a.out $VALEUR_E_x $VALEUR_E_y
# done
mpirun ./a.out 0.0 0.0
mpirun ./a.out 3.0 4.0
mpirun ./a.out 4.0 4.0
mpirun ./a.out 3.0 2.0
echo "Tests terminés. Résultats dans /data_files"
rm a.out