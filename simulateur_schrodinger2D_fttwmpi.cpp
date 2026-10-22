///////////////////////////////////////////////////////////////////////////////////////////////
//simulateur de l'équation de schrodinger: i\partial_{t}\psi=-\frac}{1}{2}\partial_{xx}\psi+V\psi
///////////////////////////////////////////////////////////////////////////////////////////////
#include<iostream>
#include<cmath>
#include<numbers>
#include<vector>
#include<fftw3-mpi.h>
#include<complex>
#include <chrono> 
#include <fstream>
#include <mpi.h>


using namespace std;
using namespace std::literals::complex_literals;

///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// VARIABLES //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

int Nx=512; // NUMBERS OF POINT ON THE X-AXIS
int Ny=512; // NUMBERS OF POINT ON THE Y-AXIS
double Lx=40.0; // SIZE OF THE DOMAIN FOR THE X-AXIS
double Ly=40.0; // SIZE OF THE DOMAIN FOR THE Y-AXIS
double T=2.0*std::numbers::pi; // TOTAL TIME OF THE OBSERVATION
double dx=Lx/Nx; 
double dy=Ly/Ny;
double dt=0.0025;
int Nt = static_cast<int>(T/dt); 
int nombre_images=T*1.5*24;
int pas_enregistrement=Nt/nombre_images; // RECORDING FREQUANCY
double k0x = 0.0; // impulsion initiale vers la droite en +
double k0y = 1.0; // impulsion initiale vers la droite en +
double sigma = 1.0;

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// POTENTIAL ENERGY FUNCTION ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

double V(double x, double y){
    return 0.5*(x*x+y*y);
    /*double x_mur=0.0;
    double epaisseur_mur=1.0;
    double largeur_fente=(2*std::numbers::pi)/(std::sqrt(k0x*k0x+k0y*k0y));
    double ecratement=4.0;
    double fente_haut=ecratement/2;
    double fente_bas=-ecratement/2;
    double raideur = 5.0;
    double profil_mur_x = 1.0 / (1.0 + std::exp(-raideur * (epaisseur_mur/2.0 - std::abs(x - x_mur))));
    double loin_fente_haut = 1.0 / (1.0 + std::exp(-raideur * (std::abs(y - fente_haut) - largeur_fente/2.0)));
    double loin_fente_bas  = 1.0 / (1.0 + std::exp(-raideur * (std::abs(y - fente_bas)  - largeur_fente/2.0)));
    return 1000.0 * profil_mur_x * std::min(loin_fente_haut, loin_fente_bas);*/
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

double masque_absorbant(double x, double y, double L) {
    double limite = 0.85 * (L / 2.0); // Zone de sécurité (85% de la demi-boîte)
    
    // On mesure l'éloignement maximal par rapport au centre (en X ou en Y)
    double distance_max = std::max(std::abs(x), std::abs(y));
    
    // Si on dépasse la limite de sécurité, on applique l'atténuation
    if (distance_max > limite) {
        double epaisseur_bord = (L / 2.0) - limite;
        double distance_dans_le_bord = distance_max - limite;
        
        // Renvoie un coefficient qui glisse doucement de 1.0 à 0.0 au ras du bord
        return std::cos((std::numbers::pi / 2.0) * (distance_dans_le_bord / epaisseur_bord));
    }
    
    return 1.0; // Pas d'atténuation au centre
}

// Absorption locale immédiatement derrière la source, pour éviter
// que l'onde émise vers la gauche ne s'accumule en cavité résonnante
double masque_arriere_source(double x, double x_source, double largeur = 3.0) {
    double distance = x_source - x; // positif = à gauche de la source
    if (distance <= 0.0) return 1.0;           // à droite de la source : intact
    if (distance >= largeur) return 0.0;       // au-delà : complètement éteint
    return std::cos((std::numbers::pi / 2.0) * (distance / largeur));
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

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
///////////////////////////// FAST FOURIER TRANSFORM WITH fftw3 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

class FFT2D
{
    private:
    fftw_complex *in;
    fftw_complex *out;
    fftw_plan pfft;
    fftw_plan pifft;
    int size_total_x;
    int size_total_y;

    public:
    FFT2D(const FFT2D&) = delete;             
    FFT2D& operator=(const FFT2D&) = delete;  
    ptrdiff_t alloc_local, local_n0, local_0_start; 

    FFT2D(int Nx, int Ny): size_total_x(Nx), size_total_y(Ny){
        alloc_local = fftw_mpi_local_size_2d(size_total_x, size_total_y, MPI_COMM_WORLD, &local_n0, &local_0_start);
        in = fftw_alloc_complex(alloc_local);
        out = fftw_alloc_complex(alloc_local);
        pfft = fftw_mpi_plan_dft_2d(size_total_x, size_total_y, in, out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
        pifft = fftw_mpi_plan_dft_2d(size_total_x, size_total_y, in, out, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    // Calcule la FFT directe (équivalent à np.fft.fft)
    // input : vecteur de taille Nx contenant les données spatiales psi(xn)
    // output : vecteur de taille Nx qui recevra les coefficients c_ktilde
    void appliquer_fft(const std::vector<std::complex<double>>& input, std::vector<std::complex<double>>& output) {
        // Copie ultra-rapide par ligne
        std::complex<double>* in_ptr = reinterpret_cast<std::complex<double>*>(in);
        for (int i = 0; i < local_n0; ++i) {
            for (int j = 0; j< size_total_y; j++){
                in_ptr[i*size_total_y+j]= input[(i)*size_total_y+j];

            }
        }
        fftw_execute(pfft);
        // Récupération ultra-rapide par ligne
        std::complex<double>* out_ptr = reinterpret_cast<std::complex<double>*>(out);
        for (int i = 0; i < local_n0; ++i) {
            for (int j = 0; j< size_total_y; j++){
                output[(i)*size_total_y+j]=out_ptr[i*size_total_y+j];

            }
        }
    }
    // Calcule la FFT inverse (équivalent à np.fft.ifft)
    // input : coefficients c_ktilde
    // output : données retournées dans l'espace réel psi(xn)
    void appliquer_ifft(const std::vector<std::complex<double>>& input, std::vector<std::complex<double>>& output) {
        std::complex<double>* in_ptr = reinterpret_cast<std::complex<double>*>(in);
        for (int i = 0; i < local_n0; ++i) {
            for (int j=0; j<size_total_y; j++){
                in_ptr[i*size_total_y+j]= input[(i)*size_total_y+j];
            }
        }

        fftw_execute(pifft);

        double norm = static_cast<double>(size_total_x * size_total_y);
        std::complex<double>* out_ptr = reinterpret_cast<std::complex<double>*>(out);
        for (int i = 0; i < local_n0; ++i) {
            for (int j = 0; j< size_total_y; j++){
                output[(i)*size_total_y+j]=out_ptr[i*size_total_y+j];
                output[(i)*size_total_y+j] /= norm;
            }
        }
    }
    ~FFT2D(){
        fftw_destroy_plan(pifft);
        fftw_destroy_plan(pfft);
        fftw_free(in);
        fftw_free(out);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// INITIAL CONDITION /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

std::complex<double> condition_initiale(double x, double y, double sigma, double k0x, double k0y) {
    double x_centre = 2.0; // -gauche; + droite
    double y_centre = 0.0;
    double enveloppe = std::exp(-((x - x_centre) * (x - x_centre)+(y-y_centre)*(y-y_centre)) / (2.0 * sigma * sigma));
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
        if (pas_enregistrement<1){
        pas_enregistrement=1;
        }
        FFT2D methodefft2d(Nx, Ny);
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////// GLOBAL VECTORS ///////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<double>lx(Nx);
        std::vector<double>ly(Ny);
        std::vector<int>k_physique_x(Nx);
        std::vector<int>k_physique_y(Ny);
        std::vector<std::complex<double>>Potentiel_V_global(Nx*Ny);
        std::vector<std::complex<double>>Matrice_k_global(Nx*Ny);
        std::vector<std::complex<double>>lf_global(Nx * Ny);
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////// LOCAL VECTORS ////////////////////////////////////////
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
            counts[i] = all_local_n0[i] * Ny;
    
            // Le déplacement correspond au cumul des éléments des CPU précédents
            displs[i] = total_elements;
    
            total_elements += counts[i];
        }

        std::cout << "My rank is: "<< myRank << " local_n0: "<< methodefft2d.local_n0 <<" local_0_start: "<< methodefft2d.local_0_start <<" alloc_local: "<< methodefft2d.alloc_local;
        std::cout << "=======================================================\n";
            auto debut_fft = std::chrono::high_resolution_clock::now();
        if (myRank == 0) {
            //On change les k pour le repliment du spectre
            for (int k = 0; k < Nx; k++) {
                k<Nx/2 ? k_physique_x[k]=k : k_physique_x[k] = (k - Nx);
            }

            for (int k = 0; k < Ny; k++) {
                k<Ny/2 ? k_physique_y[k]=k : k_physique_y[k] = (k - Ny);
                }
            
            //maillage spatial et condition initial
            for (int n = 0; n < Nx; n++) {
                double k_spectre_x=k_physique_x[n]*(2*std::numbers::pi/Lx);
                double facteur_x=0.5*k_spectre_x*k_spectre_x;
                lx[n] = -Lx/2.0+n*dx;
                for (int m=0; m< Ny; m++){
                    double k_spectre_y=k_physique_y[m]*(2*std::numbers::pi/Ly);
                    double facteur_y=0.5*k_spectre_y*k_spectre_y;
                    ly[m] = -Ly/2.0+m*dy;
                    Matrice_k_global[n * Ny + m]=std::exp(-1i*(facteur_x+facteur_y)*dt); // WE STOCK DATA OF THE K FOR THE CALCULUS OF THE LAPLACIEN WITH THE FFT
                    lf_global[n * Ny + m] = condition_initiale(lx[n], ly[m], sigma, k0x, k0y);
                    Potentiel_V_global[n * Ny + m]= std::exp(-1i * V(lx[n], ly[m]) * dt/2.0); // WE STOCK DATA OF THE POTENTIAL IN A MATRIX 
                }
            }
        }

        MPI_Scatterv(lf_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                lf_local.data(), methodefft2d.local_n0 * Ny, MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
        MPI_Scatterv(Matrice_k_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                Matrice_k_local.data(), methodefft2d.local_n0 * Ny, MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
        MPI_Scatterv(Potentiel_V_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                Potentiel_V_local.data(), methodefft2d.local_n0 * Ny, MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

        std::ofstream fichier_donnees("evolution_schrodinger2D_mpi.bin", std::ios::binary);
        for (int t=0; t<Nt; t++){
            if (t % pas_enregistrement == 0) {
                MPI_Gatherv(lf_local.data(), methodefft2d.local_n0 * Ny, MPI_C_DOUBLE_COMPLEX,
                lf_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
                if (myRank==0){
                    for (int n = 0; n < Nx; n++) {
                        for (int m = 0; m < Ny; m++) {
                            double densite = std::norm(lf_global[n * Ny + m]);
                            fichier_donnees.write(reinterpret_cast<char*>(&densite), sizeof(double));
                        }
                    }
                }
            }
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < Ny; m++) {
                    lf_local[i * Ny + m] *=  Potentiel_V_local[i*Ny+m];
                }
            }

            // 1. FFT locale sur chaque CPU
            methodefft2d.appliquer_fft(lf_local, ck_tilde_local);

            // 2. Filtrage dans l'espace K (Chaque CPU traite ses lignes locales)
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < Ny; m++) {
                    d2psi_local[i * Ny + m] = ck_tilde_local[i * Ny + m]*Matrice_k_local[i*Ny+m];
                }
            }

            // 3. IFFT locale sur chaque CPU
            methodefft2d.appliquer_ifft(d2psi_local, lf_local); 
            for (int i = 0; i < methodefft2d.local_n0; i++) {
                for (int m = 0; m < Ny; m++) {
                    lf_local[i * Ny + m] *=  Potentiel_V_local[i*Ny+m];
                }
            }
        }
        MPI_Gatherv(lf_local.data(), methodefft2d.local_n0 * Ny, MPI_C_DOUBLE_COMPLEX,
                lf_global.data(), counts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
        auto fin_fft = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> temps_total = fin_fft - debut_fft;
        fichier_donnees.close();
        
        if (myRank == 0) {
            double somme_carres = 0.0;
            double t_final = Nt * dt; 

            // 1. ÉTAPE DE CALCUL DE L'ERREUR EN CORRIGEANT LA PHASE GLOBALE DE STRANG
            for (int n = 0; n < Nx; ++n) {
                double x = lx[n]; 
                for (int m = 0; m < Ny; ++m) {
                    double y = ly[m]; 
                    
                    // Réalignement de la phase finale de Strang (dt/2)
                    std::complex<double> correction_phase = std::exp(1i * V(x, y) * (dt / 2.0));
                    std::complex<double> psi_num_corrige = lf_global[n * Ny + m] * correction_phase;
                    
                    // Évaluation de la solution théorique (sans le préfacteur comme validé)
                    std::complex<double> psi_th = solution_theorique(x, y, t_final, sigma, k0x, k0y);
                    
                    std::complex<double> diff = psi_num_corrige - psi_th;
                    somme_carres += std::norm(diff);
                }
            }
            
            // Intégration spatiale pour obtenir la norme L2 brute
            double erreur_L2 = std::sqrt(somme_carres * dx * dy);

            // --- RESTAURATION DE L'AFFICHAGE BRUT DE BENCHMARK ---
            std::cout << "=======================================================\n";
            std::cout << "               BENCHMARK DE PERFORMANCE                \n";
            std::cout << "=======================================================\n";
            std::cout << "Temps total pour " << Nt << " pas de temps : " << temps_total.count() << " ms\n";
            std::cout << "Temps moyen par itération : " << temps_total.count() / Nt << " ms\n";
            std::cout << "=======================================================\n";
            std::cout << "Nombre de points : " << static_cast<long long>(Nx) * Ny * Nt << "\n";
            std::cout << "=======================================================\n";
            std::cout << "Erreur en norme L2: " << std::scientific << std::setprecision(2) << erreur_L2 << std::endl;
        }
    }
        fftw_mpi_cleanup();
        MPI_Finalize();
        return 0;
}
