///////////////////////////////////////////////////////////////////////////////////////////////
//simulateur de l'équation de schrodinger: i\partial_{t}\psi=-\frac}{1}{2}\partial_{xx}\psi+V\psi
///////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// FENTES DE YOUNG //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
#include "Common.hpp"
#include "Parametres.hpp"
#include "Potentiel.hpp"
#include "fftw-mpi.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

Parametres param;

std::complex<double> solution_theorique(double x, double y, double t, double sigma, double k0x, double k0y){
    double x0 = 2.0; 
    double y0 = 0.0;
    double xt = x0 * std::cos(t) + k0x * std::sin(t);
    double yt = y0 * std::cos(t) + k0y * std::sin(t);
    double kxt = -x0 * std::sin(t) + k0x * std::cos(t);
    double kyt = -y0 * std::sin(t) + k0y * std::cos(t);
    double phase_globale = -t;
    double phase_espace = kxt * (x - xt) + kyt * (y - yt);
    double phase_action = 0.5 * (kxt*xt + kyt*yt - k0x*x0 - k0y*y0);
    double gaussienne = std::exp(-((x - xt)*(x - xt) + (y - yt)*(y - yt)) / (2.0 * sigma * sigma));
    return gaussienne * std::exp(1i * (phase_espace + phase_action + phase_globale));
}

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// INITIAL CONDITION /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

std::complex<double> condition_initiale(double x, double y, double sigma, double k0x, double k0y) {
    double enveloppe = std::exp(-((x - param.x_centre) * (x - param.x_centre)+(y-param.y_centre)*(y-param.y_centre)) / (2.0 * sigma * sigma));
    std::complex<double> phasex = std::exp(1i * k0x * x);
    std::complex<double> phasey = std::exp(1i * k0y * y);    
    return enveloppe * phasex* phasey;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// FUNCTIONS FOR TEST ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

std::complex<double> Gaussienne_en_0(const double x, const double y){
    return std::exp(-(x*x+y*y)/2.0);
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// MAIN ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    {
        fftw_mpi_init();
        int myRank;
        int nbTask;
        MPI_Comm_size(MPI_COMM_WORLD, &nbTask);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        if (param.pas_enregistrement<1){
            param.pas_enregistrement=1;
        }
        FFT2D methodefft2d(param.Nx, param.Ny);
///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// GLOBAL VECTORS ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<double>lx(param.Nx);
        std::vector<double>ly(param.Ny);
        std::vector<int>k_physique_x(param.Nx);
        std::vector<int>k_physique_y(param.Ny);
        std::vector<std::complex<double>>Potentiel_V_global(param.Nx*param.Ny);
        std::vector<std::complex<double>>Matrice_k_global(param.Nx*param.Ny);
        std::vector<std::complex<double>>lf_global(param.Nx * param.Ny);
        std::ofstream fichier_donnees("evolution_schrodinger2D_mpi.bin", std::ios::binary);
        std::ofstream fichier_potentiel("potentiel_2D.bin", std::ios::binary);

///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// LOCAL VECTORS //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<std::complex<double>> lf_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> Potentiel_V_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> Matrice_k_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> ck_tilde_local(methodefft2d.alloc_local);
        std::vector<std::complex<double>> d2psi_local(methodefft2d.alloc_local);
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

    
        // 1. Chaque CPU partage son 'local_n0' avec tous les autres
        std::vector<int> all_local_n0(nbTask);
        int local_n0_int = static_cast<int>(methodefft2d.local_n0);

        MPI_Allgather(&local_n0_int, 1, MPI_INT, all_local_n0.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // 2. Allocation des vecteurs counts et displs (utilisés par Scatterv/Gatherv)
        std::vector<int> counts(nbTask);
        std::vector<int> displs(nbTask);

        int total_elements = 0;
        for (int i = 0; i < nbTask; ++i) {
            // Le nombre d'éléments pour le CPU 'i' est sa quantité de lignes multipliée par Ny
            counts[i] = all_local_n0[i] * param.Ny;
    
            // Le déplacement correspond au cumul des éléments des CPU précédents
            displs[i] = total_elements;
    
            total_elements += counts[i];
        }

        std::cout << "My rank is: "<< myRank << " local_n0: "<< methodefft2d.local_n0 <<" local_0_start: "<< methodefft2d.local_0_start <<" alloc_local: "<< methodefft2d.alloc_local;
        std::cout << "=======================================================\n";
            auto debut_fft = std::chrono::high_resolution_clock::now();
        if (myRank == 0) {
            std::ofstream fichier_donnees("evolution_schrodinger2D_mpi.bin", std::ios::binary);
            //On change les k pour le repliment du spectre
            for (int k = 0; k < param.Nx; k++) {
                k<param.Nx/2 ? k_physique_x[k]=k : k_physique_x[k] = (k - param.Nx);
            }

            for (int k = 0; k < param.Ny; k++) {
                k<param.Ny/2 ? k_physique_y[k]=k : k_physique_y[k] = (k - param.Ny);
                }
            
            //maillage spatial et condition initial
            for (int n = 0; n < param.Nx; n++) {
                double k_spectre_x=k_physique_x[n]*(2*std::numbers::pi/param.Lx);
                double facteur_x=0.5*k_spectre_x*k_spectre_x;
                lx[n] = -param.Lx/2.0+n*param.dx;
                for (int m=0; m< param.Ny; m++){
                    double k_spectre_y=k_physique_y[m]*(2*std::numbers::pi/param.Ly);
                    double facteur_y=0.5*k_spectre_y*k_spectre_y;
                    ly[m] = -param.Ly/2.0+m*param.dy;
                    Matrice_k_global[n * param.Ny + m]=std::exp(-1i*(facteur_x+facteur_y)*param.dt); // WE STOCK DATA OF THE K FOR THE CALCULUS OF THE LAPLACIEN WITH THE FFT
                    lf_global[n * param.Ny + m] = condition_initiale(lx[n], ly[m], param.sigma, param.k0x, param.k0y);
                    Potentiel_V_global[n * param.Ny + m]= std::exp(-1i * V(lx[n], ly[m]) * param.dt/2.0); // WE STOCK DATA OF THE POTENTIAL IN A MATRIX
                    double potentiel_norme = std::norm(V(lx[n],ly[m]));
                    fichier_potentiel.write(reinterpret_cast<char*>(&potentiel_norme), sizeof(double));
                }
            }
        }

        MPI_Scatterv(lf_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                lf_local.data(), methodefft2d.local_n0 * param.Ny, MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
        MPI_Scatterv(Matrice_k_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                Matrice_k_local.data(), methodefft2d.local_n0 * param.Ny, MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
        MPI_Scatterv(Potentiel_V_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                Potentiel_V_local.data(), methodefft2d.local_n0 * param.Ny, MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);


///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// TIME LOOP /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////        
        for (int t=0; t<param.Nt; t++){
            if (t % param.pas_enregistrement == 0) {
                MPI_Gatherv(lf_local.data(), methodefft2d.local_n0 * param.Ny, MPI_C_DOUBLE_COMPLEX,
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
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < param.Ny; m++) {
                    lf_local[i * param.Ny + m] *=  Potentiel_V_local[i*param.Ny+m];
                }
            }

            // 1. FFT locale sur chaque CPU
            methodefft2d.appliquer_fft(lf_local, ck_tilde_local);

            // 2. Filtrage dans l'espace K (Chaque CPU traite ses lignes locales)
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < param.Ny; m++) {
                    d2psi_local[i * param.Ny + m] = ck_tilde_local[i * param.Ny + m]*Matrice_k_local[i*param.Ny+m];
                }
            }

            // 3. IFFT locale sur chaque CPU
            methodefft2d.appliquer_ifft(d2psi_local, lf_local); 
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < param.Ny; m++) {
                    lf_local[i * param.Ny + m] *=  Potentiel_V_local[i*param.Ny+m];
                }
            }
        }

        MPI_Gatherv(lf_local.data(), methodefft2d.local_n0 * param.Ny, MPI_C_DOUBLE_COMPLEX,
                lf_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
        auto fin_fft = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> temps_total = fin_fft - debut_fft;
        fichier_donnees.close();
        fichier_potentiel.close();

        if (myRank == 0) {
            double somme_carres = 0.0;
            double t_final = param.Nt * param.dt; 
            double proba_finale=0.0;
            double proba_initiale= 0.0;

            // 1. ÉTAPE DE CALCUL DE L'ERREUR EN CORRIGEANT LA PHASE GLOBALE DE STRANG
            for (int n = 0; n < param.Nx; ++n) {
                double x = lx[n]; 
                for (int m = 0; m < param.Ny; ++m) {
                    double y = ly[m]; 
                    
                    // Évaluation de la solution théorique (sans le préfacteur comme validé)
                    std::complex<double> psi_th = solution_theorique(x, y, t_final, param.sigma, param.k0x, param.k0y);
                    
                    std::complex<double> diff = lf_global[n * param.Ny + m] - psi_th;
                    somme_carres += std::norm(diff);
                    proba_finale+= std::norm(lf_global[n * param.Ny + m]);
                    proba_initiale+= std::norm(condition_initiale(lx[n], ly[m], param.sigma, param.k0x, param.k0y));
                }
            }
            
            // Intégration spatiale pour obtenir la norme L2 brute
            double erreur_L2 = std::sqrt(somme_carres * param.dx * param.dy);
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
            std::cout << "Erreur en norme L2: " << std::scientific << std::setprecision(2) << erreur_L2 << std::endl;
            std::cout << "=======================================================\n";
            std::cout << "Proba fianle: " << proba_finale <<std::endl;
            std::cout << "Proba initiale: " << proba_initiale <<std::endl;
        }
    }
        fftw_mpi_cleanup();
        MPI_Finalize();
        return 0;
}
