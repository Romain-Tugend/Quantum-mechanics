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
std::ofstream fichier_donnees;
std::ofstream fichier_potentiel;

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
        if (argc >= 2) {
            param.W = atof(argv[1]); // Amplitude W du désordre
        }
        if (myRank == 0) {
            std::cout << "Simulation Anderson avec W = " << param.W << std::endl;
            std::string suffixe = "_W_" + std::to_string(param.W) + ".bin";
            fichier_potentiel.open("data_files/potentiel_2D" + suffixe, std::ios::binary);
            fichier_donnees.open("data_files/evolution_schrodinger2D_mpi" + suffixe, std::ios::binary);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double time_init = MPI_Wtime();
        if (param.pas_enregistrement<1){
            param.pas_enregistrement=1;
        }
        FFT2D methodefft2d(param.Nx, param.Ny); 
        // WE BUILD DATA FOR K THAT WILL BE USE FOR LAPLACIEN CALCULUS 
        mpi_build_global_vectors();
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// VECTORS /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<std::complex<double>> Mesh_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> Potentiel_V_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> Matrice_k_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> Fourier_local(methodefft2d.alloc_local);
        
        counts.resize(nbTask);
        displs.resize(nbTask);
        // BUILD DATA FOR GATHERV WHEN WE NEED THE GLOBAL SOLUTION
        mpi_vector_coms_build(methodefft2d.local_n0, counts, displs);
        
        // WE BUILD LOCAL VECTORS FOR FFT 
        mpi_build_local_vectors(methodefft2d.local_n0, methodefft2d.local_0_start, Mesh_local, Potentiel_V_local, Matrice_k_local);
        

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// TIME LOOP /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////        
        for (int t=0; t<param.Nt; t++){
            // WE COMPUTE THE PSEUDO-SPECTRAL METHODE FOR DISCRETISATION WITH GALERKIN
            // FOR A NEW TIME STEP WE DO ....
            
            // WRITTING THE SOLUTION FOR SPECIAL TIMES TO GET 36 FPS
            if (t % param.pas_enregistrement == 0) {
                mpi_file_writting(Mesh_local, methodefft2d.local_n0, counts, displs);
            }
            
            // 1.) POTENTIAL CALCULUS
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < param.Ny; m++) {
                    Mesh_local[i * param.Ny + m] *=  Potentiel_V_local[i*param.Ny+m];
                }
            }
            // 2.) LAPLACIEN CALCULS WITH FFT
            // FFT TO GET INTO FOURRIER SPACE
            methodefft2d.appliquer_fft(Mesh_local, Fourier_local);
            // THE LAPLACIEN IS JUST A MULTIPLICATION IN FOURRIER SPACE
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < param.Ny; m++) {
                    Fourier_local[i * param.Ny + m] *= Matrice_k_local[i*param.Ny+m];
                }
            }
            // GET BACK INTO REAL SPACE WITH IFFT
            methodefft2d.appliquer_ifft(Fourier_local, Mesh_local); 
        }
        
        MPI_Gatherv(Mesh_local.data(), methodefft2d.local_n0 * param.Ny, MPI_C_DOUBLE_COMPLEX,
                lf_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
        fichier_donnees.close();
        MPI_Barrier(MPI_COMM_WORLD);
        double time_end = MPI_Wtime();
        if (myRank == 0) {
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
            std::cout << "Proba fianle: " << proba_finale <<std::endl;
            std::cout << "Proba initiale: " << proba_initiale <<std::endl;
        }
    }
        fftw_mpi_cleanup();
        MPI_Finalize();
        return 0;
}
