//simulateur de l'équation de schrodinger: i\partial_{t}\psi=-\frac}{1}{2}\partial_{xx}\psi+V\psi
#include <iostream>
#include <cmath>
#include <numbers>
#include <vector>
#include <fftw3.h>
#include <complex>
#include <chrono> 
#include <fstream>
#include <mpi.h>
#include <string>

using namespace std;
using namespace std::literals::complex_literals;


///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// VARIABLES //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

int Nx=1024; // NUMBERS OF POINT ON THE X-AXIS
int Ny=1024; // NUMBERS OF POINT ON THE Y-AXIS
double Lx=40.0; // SIZE OF THE DOMAIN FOR THE X-AXIS
double Ly=40.0; // SIZE OF THE DOMAIN FOR THE Y-AXIS
double T=15.0; // TOTAL TIME OF THE OBSERVATION
double dx=Lx/Nx; 
double dy=Ly/Ny;
double dt=0.01;
int Nt = static_cast<int>(T/dt); 
int nombre_images=T*1.5*24;
int pas_enregistrement=Nt/nombre_images; // RECORDING FREQUANCY
double k0x = 5.0; // impulsion initiale vers la droite en +
double k0y = 0.0; // impulsion initiale vers la droite en +

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// POTENTIAL ENERGY FUNCTION ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

double V(double x, double y){
    //double n=std::sqrt(x*x+y*y);
    double x_mur=0.0;
    double epaisseur_mur=1.0;
    double largeur_fente=(2*std::numbers::pi)/(std::sqrt(k0x*k0x+k0y*k0y));
    double ecratement=4.0;
    double fente_haut=ecratement/2;
    double fente_bas=-ecratement/2;
    double raideur = 5.0;
    double profil_mur_x = 1.0 / (1.0 + std::exp(-raideur * (epaisseur_mur/2.0 - std::abs(x - x_mur))));
    double loin_fente_haut = 1.0 / (1.0 + std::exp(-raideur * (std::abs(y - fente_haut) - largeur_fente/2.0)));
    double loin_fente_bas  = 1.0 / (1.0 + std::exp(-raideur * (std::abs(y - fente_bas)  - largeur_fente/2.0)));

    return 1000.0 * profil_mur_x * std::min(loin_fente_haut, loin_fente_bas);
    //return 0.5*(x*x+y*y);
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

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// FAST FOURIER TRANSFORM WITH fftw3 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

class FFT2D
{
    private:
    fftw_complex *in= nullptr;
    fftw_complex *out= nullptr;
    fftw_plan pfft= nullptr;
    fftw_plan pifft= nullptr;
    int ny_local=0;
    int nx_local=0;
    int taille_local=0;

    public:
    FFT2D(const FFT2D&) = delete;             
    FFT2D& operator=(const FFT2D&) = delete;   

    FFT2D(int Nx,int Ny): nx_local(Nx), ny_local(Ny){
        taille_local=nx_local*ny_local;
        in = (fftw_complex*) fftw_malloc(taille_local * sizeof(fftw_complex));
        out = (fftw_complex*) fftw_malloc(taille_local * sizeof(fftw_complex));
        if (in == nullptr || out == nullptr) {
            std::cerr << "Erreur : Échec d'allocation mémoire FFTW !" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        pfft = fftw_plan_dft_2d(nx_local, ny_local, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        pifft = fftw_plan_dft_2d(nx_local, ny_local, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    // Calcule la FFT directe (équivalent à np.fft.fft)
    // input : vecteur de taille Nx*Ny contenant les données spatiales psi(xn)
    // output : vecteur de taille Nx*Ny qui recevra les coefficients c_ktilde
    void appliquer_fft(const std::vector<std::complex<double>>& input, std::vector<std::complex<double>>& output) {
        int limite = std::min({taille_local, (int)input.size(), (int)output.size()});
        for (int i = 0; i < limite; ++i) {
            in[i][0] = input[i].real(); 
            in[i][1] = input[i].imag(); 
            }
        fftw_execute(pfft);
        for (int i = 0; i < limite; ++i) {
            output[i] = std::complex<double>(out[i][0], out[i][1]);
        }
    }
    // Calcule la FFT inverse (équivalent à np.fft.ifft)
    // input : coefficients c_ktilde
    // output : données retournées dans l'espace réel psi(xn)
    void appliquer_ifft(const std::vector<std::complex<double>>& input, std::vector<std::complex<double>>& output) {
        int limite = std::min({taille_local, (int)input.size(), (int)output.size()});
        for (int i = 0; i < limite; ++i) {
            in[i][0] = input[i].real(); 
            in[i][1] = input[i].imag(); 
            }
        fftw_execute(pifft);
        double norm = static_cast<double>(Nx * Ny);
        for (int i = 0; i < limite; ++i) {
            output[i] = std::complex<double>(out[i][0], out[i][1])/norm;
        }
    }

    ~FFT2D(){
        if (pifft != nullptr) {
            fftw_destroy_plan(pifft);
            pifft = nullptr;
        }
        if (pfft != nullptr) {
            fftw_destroy_plan(pfft);
            pfft = nullptr;
        }
        if (in != nullptr) {
            fftw_free(in);
            in = nullptr;
        }
        if (out != nullptr) {
            fftw_free(out);
            out = nullptr;
        }
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// INITIAL CONDITION /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

std::complex<double> condition_initiale(double x, double y, double sigma, double k0x, double k0y) {
    double x_centre = -5.0; // - gauche; + droite
    double y_centre = -7.0; //- bas; + haut
    double enveloppe = std::exp(-((x - x_centre) * (x - x_centre)+(y-y_centre)*(y-y_centre)) / (2.0 * sigma * sigma));
    std::complex<double> phasex = std::exp(1i * k0x * x);
    std::complex<double> phasey = std::exp(1i * k0y * y);    
    return enveloppe * phasex* phasey;
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// MAIN ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    {
    int myRank;
    int nbTask;
    MPI_Comm_size(MPI_COMM_WORLD, &nbTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (pas_enregistrement<1){
    pas_enregistrement=1;
    }
    int m_start= myRank*((int)(Ny/nbTask)+1);
    int m_end= std::min((myRank+1)*((int)(Ny/nbTask)+1),Ny-1);
    int m_size =m_end-m_start+1;

    std::vector<double>lx(Nx);
    std::vector<double>ly(Ny);
    std::vector<std::complex<double>>lf(Nx*m_size, 0.0);
    std::vector<std::complex<double>>ck_tilde(Nx*m_size, 0.0);
    std::vector<std::complex<double>>d2psi_k(Nx*m_size, 0.0);
    std::vector<std::complex<double>>lf_global(Nx*Ny, 0.0);
    std::vector<int>k_physique_x(Nx);
    std::vector<int>k_physique_y(Ny);
    std::string nom_fichier = "evolution_schrodinger2D_mpi_cpu" +std::to_string(myRank) + ".bin";
    std::ofstream fichier_donnees(nom_fichier, std::ios::binary);
    FFT2D methodefft2d(Nx, Ny);
    
    //On change les k pour le repliment du spectre
    for (int k = 0; k < Nx; k++) {
        if (k<=Nx/2){
            k_physique_x[k]=k;
        }
        else{
            k_physique_x[k] = (k - Nx);
        }
    }

    for (int k = 0; k < Ny; k++) {
        if (k<=Ny/2){
            k_physique_y[k]=k;
        }
        else{
            k_physique_y[k] = (k - Ny);
        }
    }
    // --- 1. Chronométrage ---
    using clock = std::chrono::steady_clock;
    auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    
    // Paramètres du paquet d'ondes gaussien
    double sigma = 0.8;
    //maillage spatial et condition initial
    for (int n = 0; n < Nx; n++) {
        lx[n] = -Lx/2.0+n*dx;
        for (int m=0; m< m_size; m++){
            ly[m+m_start] = -Ly/2.0+m*dy;
            lf[n + m_size*m] = condition_initiale(lx[n], ly[m+m_start], sigma, k0x, k0y);
        }
    }
    
    auto boucle_temporelle_debut = clock::now();
    for (int t=0; t<Nt; t++){
        /*
        // =================================================================
        // AJOUT : SOURCE CONTINUE (Éclairage Laser)
        // =================================================================
        int n_source = static_cast<int>((-14.0+Lx/2.0)/dx);
        // Fréquence d'oscillation temporelle associée à l'énergie cinétique E = 0.5 * k^2
        double omega = 0.5 * (k0x * k0x + k0y * k0y); 
        
        for (int m = 0; m < Ny; m++) {
            // Forme spatiale de la source sur l'axe Y (une grosse gaussienne qui arrose le mur)
            double sigma_source = 5.0; 
            double enveloppe_y = std::exp(-(ly[m] * ly[m]) / (2.0 * sigma_source * sigma_source));
        
            // On force la valeur de la fonction d'onde avec sa phase temporelle exp(-i * omega * t)
            lf[n_source][m] = enveloppe_y * std::exp(-1i * omega * (t * dt));
        }
        // =================================================================
        */
        if (t % pas_enregistrement == 0) {
            for (int n = 0; n < Nx; n++) {
                for (int m = m_start; m <=m_end ; m++) {
                    double densite = std::norm(lf[n + Ny*m]);
                    fichier_donnees.write(reinterpret_cast<char*>(&densite), sizeof(double));
                }
            }
        }

        for (int n = 0; n < Nx; n++){
            for (int m=0; m < m_size; m++ ){
                lf[n + m_size*m] = lf[n + m_size*m] * std::exp(-1i * V(lx[n], ly[m+m_start]) * dt);
                lf[n + m_size*m] = lf[n + m_size*m] * masque_absorbant(lx[n], ly[m+m_start], Lx);
                /*
                if (n!=n_source){
                    lf[n][m] = lf[n][m]*masque_absorbant(lx[n], ly[m], Lx);
                }
                if (n < n_source){
                    lf[n][m] = lf[n][m] * masque_arriere_source(lx[n], lx[n_source]);
                }
                */
            }
        }

        methodefft2d.appliquer_fft(lf,ck_tilde);

        for (int k=0; k<Nx; k++){
            double k_spectre_x=k_physique_x[k]*(2*std::numbers::pi/Lx);
            double facteur_x=k_spectre_x*k_spectre_x;
            for (int h=0; h< m_size; h++){
                double k_spectre_y=k_physique_y[h+m_start]*(2*std::numbers::pi/Ly);
                double facteur_y=0.5*k_spectre_y*k_spectre_y;
                d2psi_k[k + m_size*h]= ck_tilde[k + m_size*h]*std::exp(-1i*(facteur_x+facteur_y)*dt);
            }
        }

        methodefft2d.appliquer_ifft(d2psi_k,lf);

    }
    auto boucle_temporelle_fin = clock::now();
    double temps_partiel= seconds(boucle_temporelle_debut, boucle_temporelle_fin);
    double temps_total=0.0;
    MPI_Reduce(&temps_partiel, &temps_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    fichier_donnees.close();
    if (myRank==0){
    std::cout << "Données sauvegardées\n";
    std::cout << "=======================================================\n";
    // --- BENCHMARK DE PERFORMANCE ---
    std::cout << "\n=======================================================\n";
    std::cout << "               BENCHMARK DE PERFORMANCE                \n";
    std::cout << "=======================================================\n";
    std::cout << "Mon rank : " << myRank << " s\n";
    std::cout << "Temps total pour " << Nt << " pas de temps : " << temps_partiel << " s\n";
    std::cout << "Temps total moyen pour " << Nt << " pas de temps : " << temps_total/nbTask << " s\n";
    std::cout << "Temps moyen par itération : " << temps_partiel / Nt << " s\n";
    std::cout << "=======================================================\n";
    std::cout << "Nombre de points : " << Nx*Ny*Nt;
    }
}
    fftw_cleanup();
    MPI_Finalize();
    return 0;
}