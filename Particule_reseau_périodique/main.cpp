///////////////////////////////////////////////////////////////////////////////////////////////
//simulateur de l'équation de schrodinger: i\partial_{t}\psi=-\frac}{1}{2}\partial_{xx}\psi+V\psi
///////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// KRONING-PENNEY /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

#include "Common.hpp"
#include "Parametres.hpp"
#include "Potentiel.hpp"
#include "fftw-mpi.hpp"
#include "coms_mpi.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

inline int myRank;
inline int nbTask;

std::ofstream fichier_donnees("evolution_schrodinger2D_mpi.bin", std::ios::binary);

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// FUNCTIONS FOR TEST ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

std::complex<double> Gaussienne_en_0(const double x, const double y){
    return std::exp(-(x*x+y*y)/2.0);
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// MAIN ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    
    MPI_Init(&argc, &argv);
    {
        fftw_mpi_init();
        MPI_Comm_size(MPI_COMM_WORLD, &nbTask);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        if (param.pas_enregistrement<1){
            param.pas_enregistrement=1;
        }
        FFT2D methodefft2d(param.Nx, param.Ny); 
        mpi_build_global_vectors();
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////// LOCAL VECTORS //////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<std::complex<double>> Mesh_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> Potentiel_V_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> Matrice_k_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> Fourier_local(methodefft2d.alloc_local);
        
        counts.resize(nbTask);
        displs.resize(nbTask);

        mpi_vector_coms_build(methodefft2d.local_n0, counts, displs);
        
        std::cout << "My rank is: "<< myRank << " local_n0: "<< methodefft2d.local_n0 <<" local_0_start: "<< methodefft2d.local_0_start <<" alloc_local: "<< methodefft2d.alloc_local;
        std::cout << "=======================================================\n";
        auto debut_fft = std::chrono::high_resolution_clock::now();
        
        mpi_build_local_vectors(methodefft2d.local_n0, methodefft2d.local_0_start, Mesh_local, Potentiel_V_local, Matrice_k_local, counts, displs);
        

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// TIME LOOP /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////        
        for (int t=0; t<param.Nt; t++){
            if (t % param.pas_enregistrement == 0) {
                mpi_file_writting(Mesh_local, methodefft2d.local_n0, counts, displs);
            }
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < param.Ny; m++) {
                    Mesh_local[i * param.Ny + m] *=  Potentiel_V_local[i*param.Ny+m];
                }
            }

            // 1. FFT locale sur chaque CPU
            methodefft2d.appliquer_fft(Mesh_local, Fourier_local);

            // 2. Filtrage dans l'espace K (Chaque CPU traite ses lignes locales)
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < param.Ny; m++) {
                    Fourier_local[i * param.Ny + m] *= Matrice_k_local[i*param.Ny+m];
                }
            }

            // 3. IFFT locale sur chaque CPU
            methodefft2d.appliquer_ifft(Fourier_local, Mesh_local); 
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < param.Ny; m++) {
                    Mesh_local[i * param.Ny + m] *=  Potentiel_V_local[i*param.Ny+m];
                }
            }
        }

        MPI_Gatherv(Mesh_local.data(), methodefft2d.local_n0 * param.Ny, MPI_C_DOUBLE_COMPLEX,
                lf_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
        auto fin_fft = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> temps_total = fin_fft - debut_fft;
        fichier_donnees.close();

        if (myRank == 0) {
            double somme_carres = 0.0;
            double t_final = param.Nt * param.dt; 
            double proba_finale=0.0;
            double proba_initiale= 0.0;

            for (int n = 0; n < param.Nx; ++n) {
                for (int m = 0; m < param.Ny; ++m) {
                    proba_finale+= std::norm(lf_global[n * param.Ny + m]);
                    proba_initiale+= std::norm(condition_initiale(lx[n], ly[m], param.sigma, param.k0x, param.k0y));
                }
            }
            proba_finale*=param.dx*param.dy;
            proba_initiale*=param.dx*param.dy;

            // --- RESTAURATION DE L'AFFICHAGE BRUT DE BENCHMARK ---
            std::cout << "=======================================================\n";
            std::cout << "               BENCHMARK DE PERFORMANCE                \n";
            std::cout << "=======================================================\n";
            std::cout << "Temps total pour " << param.Nt << " pas de temps : " << temps_total.count() << " ms\n";
            std::cout << "Temps moyen par itération : " << temps_total.count() / param.Nt << " ms\n";
            std::cout << "=======================================================\n";
            std::cout << "Nombre de points : " << static_cast<long long>(param.Nx * param.Ny * param.Nt) << "\n";
            std::cout << "=======================================================\n";
            std::cout << "Proba fianle: " << proba_finale <<std::endl;
            std::cout << "Proba initiale: " << proba_initiale <<std::endl;
        }
    }
        fftw_mpi_cleanup();
        MPI_Finalize();
        return 0;
}
