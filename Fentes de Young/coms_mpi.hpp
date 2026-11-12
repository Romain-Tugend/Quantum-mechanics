#ifndef COMS_MPI_HPP
#define COMS_MPI_HPP
#include "Common.hpp"
#include "Parametres.hpp"


extern std::vector<std::complex<double>> lf_local;
extern std::vector<std::complex<double>> lf_global;
extern std::ofstream fichier_donnees;
std::ofstream fichier_potentiel("potentiel_2D.bin", std::ios::binary);
extern int nbTask;
extern int myRank;
extern FFT2D methodefft2d;

void mpi_vector_coms_build(std::ptrdiff_t local_n0, 
    std::vector<int> &counts, 
    std::vector<int> &displs){
    // 1. Chaque CPU partage son 'local_n0' avec tous les autres
    std::vector<int> all_local_n0(nbTask);
    int local_n0_int = static_cast<int>(local_n0);
    MPI_Allgather(&local_n0_int, 1, MPI_INT, all_local_n0.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 2. Allocation des vecteurs counts et displs (utilisés par Scatterv/Gatherv)
    int total_elements = 0;
    for (int i = 0; i < nbTask; ++i) {
        // Le nombre d'éléments pour le CPU 'i' est sa quantité de lignes multipliée par Ny
        counts[i] = all_local_n0[i] * param.Ny;
        // Le déplacement correspond au cumul des éléments des CPU précédents
        displs[i] = total_elements;
        total_elements += counts[i];
    }
}

void mpi_build_global_vectors(){
    for (int k = 0; k < param.Nx; k++) {
        k<param.Nx/2 ? k_physique_x[k]=k : k_physique_x[k] = (k - param.Nx);
        k_physique_x[k]*=(2*M_PI/param.Lx);
        k_physique_x[k]*=0.5*k_physique_x[k];
        lx[k] = -param.Lx/2.0+k*param.dx;
    }
    for (int k = 0; k < param.Ny; k++) {
        k<param.Ny/2 ? k_physique_y[k]=k : k_physique_y[k] = (k - param.Ny);
        k_physique_y[k]*=(2*M_PI/param.Ly);
        k_physique_y[k]*=0.5*k_physique_y[k];
        ly[k] = -param.Ly/2.0+k*param.dy;
    }
}

void mpi_build_local_vectors(std::ptrdiff_t local_n0,
    std::ptrdiff_t local_n0_start,
    std::vector<std::complex<double>> &Mesh_local,
    std::vector<std::complex<double>> &Potentiel_V_local,
    std::vector<std::complex<double>> &Matrice_k_local,
    const std::vector<int> &counts, 
    const std::vector<int> &displs)
{
    
    for (int n = 0; n < local_n0; n++) {
        for (int m = 0; m < param.Ny; m++) {
            Matrice_k_local[n*param.Ny+m]=std::exp(-1i * (param.dt*(k_physique_x[n+local_n0_start]+k_physique_y[m])));// WE STOCK DATA OF THE K FOR THE CALCULUS OF THE LAPLACIEN WITH THE FFT
            Mesh_local[n * param.Ny + m] = 0.0;
            double x_curr = lx[n + local_n0_start];
            double y_curr = ly[m];
            Potentiel_V_local[n * param.Ny + m] = std::exp((-1i * V(x_curr, y_curr) - W(x_curr, y_curr)) * (param.dt / 2.0));
        }
    }
    if (myRank == 0) {
    for (int n = 0; n < param.Nx; n++) {
        for (int m = 0; m < param.Ny; m++) {
            double potentiel_norme = V(lx[n], ly[m]);
            fichier_potentiel.write(reinterpret_cast<char*>(&potentiel_norme), sizeof(double));
        }
    }
}
    fichier_potentiel.close();
}


void mpi_file_writting(const std::vector<std::complex<double>>& Mesh_local, 
    std::ptrdiff_t local_n0, 
    const std::vector<int> &counts, 
    const std::vector<int> &displs){
    MPI_Gatherv(Mesh_local.data(), local_n0 * param.Ny, MPI_C_DOUBLE_COMPLEX,
                lf_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
    if (myRank==0){    
        for (int n = 0; n < param.Nx; n++) {
            for (int m = 0; m < param.Ny; m++) {
                double densite = std::norm(lf_global[n * param.Ny + m]);
                fichier_donnees.write(reinterpret_cast<char*>(&densite), sizeof(double));
            }
        }
    }
}


#endif /* COMS_MPI_HPP */