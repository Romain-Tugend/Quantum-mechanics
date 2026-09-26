//simulateur de l'équation de schrodinger: i\partial_{t}\psi=-\frac}{1}{2}\partial_{xx}\psi+V\psi
#include<iostream>
#include<cmath>
#include<numbers>
#include<vector>
#include<fftw3.h>
#include<complex>
#include <chrono> 
#include <fstream>

using namespace std;
using namespace std::literals::complex_literals;

int Nx=512;
int Ny=512;
double L=40.0;
double T=15.0;
double dx=L/Nx;
double dy=L/Ny;
double dt=0.01;
int Nt = static_cast<int>(T/dt);
int nombre_images=T*1.5*24;
int pas_enregistrement=Nt/nombre_images;


double k0x = 5.0; // impulsion initiale vers la droite en +
double k0y = 0.0; // impulsion initiale vers la droite en +


//fonction potentiel
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
}

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
// Calcule la FFT directe (équivalent à np.fft.fft)
// input : vecteur de taille Nx contenant les données spatiales psi(xn)
// output : vecteur de taille Nx qui recevra les coefficients c_ktilde
void appliquer_fft(const std::vector<std::vector<std::complex<double>>>& input, std::vector<std::vector<std::complex<double>>>& output) {
    int temp_nx = input.size();
    int temp_ny = input[0].size();

    fftw_complex *in = (fftw_complex*) fftw_malloc(temp_nx * temp_ny * sizeof(fftw_complex));
    fftw_complex *out = (fftw_complex*) fftw_malloc(temp_nx * temp_ny * sizeof(fftw_complex));
    
    // Copie ultra-rapide par ligne
    std::complex<double>* in_ptr = reinterpret_cast<std::complex<double>*>(in);
    for (int i = 0; i < temp_nx; ++i) {
        std::copy(input[i].begin(), input[i].end(), in_ptr + i * temp_ny);
    }

    fftw_plan p = fftw_plan_dft_2d(temp_nx, temp_ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    // Récupération ultra-rapide par ligne
    std::complex<double>* out_ptr = reinterpret_cast<std::complex<double>*>(out);
    for (int i = 0; i < temp_nx; ++i) {
        std::copy(out_ptr + i * temp_ny, out_ptr + (i + 1) * temp_ny, output[i].begin());
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}
// Calcule la FFT inverse (équivalent à np.fft.ifft)
// input : coefficients c_ktilde
// output : données retournées dans l'espace réel psi(xn)
void appliquer_ifft(const std::vector<std::vector<std::complex<double>>>& input, std::vector<std::vector<std::complex<double>>>& output) {
    int temp_nx = input.size();
    int temp_ny = input[0].size();

    fftw_complex *in = (fftw_complex*) fftw_malloc(temp_nx * temp_ny * sizeof(fftw_complex));
    fftw_complex *out = (fftw_complex*) fftw_malloc(temp_nx * temp_ny * sizeof(fftw_complex));

    std::complex<double>* in_ptr = reinterpret_cast<std::complex<double>*>(in);
    for (int i = 0; i < temp_nx; ++i) {
        std::copy(input[i].begin(), input[i].end(), in_ptr + i * temp_ny);
    }

    fftw_plan p = fftw_plan_dft_2d(temp_nx, temp_ny, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    double norm = static_cast<double>(temp_nx * temp_ny);
    std::complex<double>* out_ptr = reinterpret_cast<std::complex<double>*>(out);
    for (int i = 0; i < temp_nx; ++i) {
        std::copy(out_ptr + i * temp_ny, out_ptr + (i + 1) * temp_ny, output[i].begin());
        // Normalisation de la ligne
        for (int j = 0; j < temp_ny; ++j) {
            output[i][j] /= norm;
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}


std::complex<double> condition_initiale(double x, double y, double sigma, double k0x, double k0y) {
    double x_centre = -12.0; // -gauche; + droite
    double y_centre = -0.0;
    double enveloppe = std::exp(-((x - x_centre) * (x - x_centre)+(y-y_centre)*(y-y_centre)) / (2.0 * sigma * sigma));
    std::complex<double> phasex = std::exp(1i * k0x * x);
    std::complex<double> phasey = std::exp(1i * k0y * y);    
    return enveloppe * phasex* phasey;
}

int main(int argc, char* argv[]){
    if (pas_enregistrement<1){
    pas_enregistrement=1;
    }
    std::vector<double>lx(Nx);
    std::vector<double>ly(Ny);
    std::vector<std::vector<std::complex<double>>>lf(Nx, std::vector<std::complex<double>>(Ny));
    std::vector<std::vector<std::complex<double>>>ck_tilde(Nx, std::vector<std::complex<double>>(Ny));
    std::vector<std::vector<std::complex<double>>>d2psi_k(Nx, std::vector<std::complex<double>>(Ny));
    std::vector<int>k_physique_x(Nx);
    std::vector<int>k_physique_y(Ny);


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
    auto debut_fft = std::chrono::high_resolution_clock::now();

    // Paramètres du paquet d'ondes gaussien
    //double sigma = 0.8;
    //maillage spatial et condition initial
    for (int n = 0; n < Nx; n++) {
        lx[n] = -L/2.0+n*dx;
        for (int m=0; m< Ny; m++){
            ly[m] = -L/2.0+m*dy;
            //lf[n][m] = condition_initiale(lx[n], ly[m], sigma, k0x, k0y);
        }
    }

    
    std::ofstream fichier_donnees("evolution_schrodinger2D.bin", std::ios::binary);

    for (int t=0; t<Nt; t++){

        // =================================================================
        // AJOUT : SOURCE CONTINUE (Éclairage Laser)
        // =================================================================
        int n_source = static_cast<int>((-14.0+L/2.0)/dx);
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
    if (t % pas_enregistrement == 0) {
        for (int n = 0; n < Nx; n++) {
            for (int m = 0; m < Ny; m++) {
                double densite = std::norm(lf[n][m]);
                fichier_donnees.write(reinterpret_cast<char*>(&densite), sizeof(double));
            }
        }
    }


        for (int n = 0; n < Nx; n++){
            for (int m=0; m < Ny; m++ ){
                lf[n][m] = lf[n][m] * std::exp(-1i * V(lx[n], ly[m]) * dt);
                if (n!=n_source){
                    lf[n][m] = lf[n][m]*masque_absorbant(lx[n], ly[m], L);
                }
                if (n < n_source){
                    lf[n][m] = lf[n][m] * masque_arriere_source(lx[n], lx[n_source]);
                }
            }
        }

        appliquer_fft(lf,ck_tilde);

        for (int k=0; k<Nx; k++){
            double k_spectre_x=k_physique_x[k]*(2*std::numbers::pi/L);
            double facteur_x=0.5*k_spectre_x*k_spectre_x;
            for (int h=0; h<Ny; h++){
                double k_spectre_y=k_physique_y[h]*(2*std::numbers::pi/L);
                double facteur_y=0.5*k_spectre_y*k_spectre_y;
                d2psi_k[k][h]= ck_tilde[k][h]*std::exp(-1i*(facteur_x+facteur_y)*dt);
            }
        }

        appliquer_ifft(d2psi_k,lf);
    }


    auto fin_fft = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> temps_total = fin_fft - debut_fft;

    fichier_donnees.close();
    std::cout << "Données sauvegardées\n";

        
    std::cout << "=======================================================\n";

    // --- BENCHMARK DE PERFORMANCE ---
    std::cout << "\n=======================================================\n";
    std::cout << "               BENCHMARK DE PERFORMANCE                \n";
    std::cout << "=======================================================\n";
    std::cout << "Temps total pour " << Nt << " pas de temps : " << temps_total.count() << " ms\n";
    std::cout << "Temps moyen par itération : " << temps_total.count() / Nt << " ms\n";
    std::cout << "=======================================================\n";
    std::cout << "Nombre de points : " << Nx*Ny*Nt;

    
    return 0;
}