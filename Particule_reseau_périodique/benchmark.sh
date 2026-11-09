#!/bin/bash

# 1. Compilation propre du projet
echo "Compilation en cours..."
make -f makefile clean
make -f makefile

# 2. Nom du fichier où seront sauvegardés les chronos
FICHIER_RES="resultats_scalabilite.txt"
rm -f $FICHIER_RES

# 3. Boucle sur le nombre de cœurs (processus MPI)
for COEURS in 1 2 4 8
do
    echo "==========================================" >> $FICHIER_RES
    echo "Exécution avec $COEURS processus MPI" >> $FICHIER_RES
    echo "==========================================" >> $FICHIER_RES
    
    # Lancement du calcul et redirection de la sortie dans le fichier texte
    mpirun -np $COEURS ./a.out >> $FICHIER_RES
done

echo "Tests terminés. Résultats dans $FICHIER_RES"
