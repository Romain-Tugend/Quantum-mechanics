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
    if (argc >= 3) {
        param.Nx = std::atoi(argv[1]);
        param.Ny = std::atoi(argv[2]);
        param.dx = param.Lx / param.Nx;
        param.dy = param.Ly / param.Ny;
        lx.resize(param.Nx);
        ly.resize(param.Ny);
        k_physique_x.resize(param.Nx);
        k_physique_y.resize(param.Ny);
    }
    MPI_Init(&argc, &argv);
    {
        fftw_mpi_init();
        MPI_Comm_size(MPI_COMM_WORLD, &nbTask);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_init = MPI_Wtime();
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
        
        mpi_build_local_vectors(methodefft2d.local_n0, methodefft2d.local_0_start, Mesh_local, Potentiel_V_local, Matrice_k_local, counts, displs);
        

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// TIME LOOP /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////        
        for (int t=0; t<param.Nt; t++){
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

        MPI_Barrier(MPI_COMM_WORLD);
        double time_end = MPI_Wtime();

        if (myRank == 0) {
            std::cout << "=======================================================\n";
            std::cout << "                       STATS MPI                       \n";
            std::cout << "=======================================================\n";
            std::cout << "Nombre de CPUs : " << nbTask << "\n";
            std::cout << "Nombre de points totaux : " << static_cast<long long>(param.Nx * param.Ny * param.Nt) << "\n";
            std::cout << "Nombre de points spatiaux : " << static_cast<long long>(param.Nx * param.Ny) << "\n";
            std::cout << "Nombre de points admissible au MPI : " << static_cast<long long>(param.Nx) << "\n";
            std::cout << "Temps total pour " << param.Nt << " pas de temps : " << time_end - time_init << " s\n";
            std::cout << "Temps moyen par itération : " << (time_end - time_init) / param.Nt << " s\n";
            std::cout << "=======================================================\n";
        }
    }
        fftw_mpi_cleanup();
        MPI_Finalize();
        return 0;
}
