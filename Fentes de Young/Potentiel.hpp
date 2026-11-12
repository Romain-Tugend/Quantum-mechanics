#ifndef POTENTIEL_HPP
#define POTENTIEL_HPP

#include "Common.hpp"
#include "Parametres.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// POTENTIAL ENERGY FUNCTION ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

double V(double x, double y) {
    
    const double x_mur = 0.0;
    const double epaisseur_mur = 0.5;

    const double ecratement = 2.0;
    const double largeur_fente = 1.2;

    const double V0 = 100.0;

    const double fente_haut =  ecratement / 2.0;
    const double fente_bas  = -ecratement / 2.0;

    if (x<=(-param.Lx/2 +2)){
        return V0/2.0;
    }
    

    // En dehors du mur : potentiel nul
    if (std::abs(x - x_mur) > epaisseur_mur / 2.0)
        return 0.0;

    // Dans le mur : potentiel nul dans les deux fentes
    bool dans_fente_haut =
        std::abs(y - fente_haut) < largeur_fente / 2.0;

    bool dans_fente_bas =
        std::abs(y - fente_bas) < largeur_fente / 2.0;

    if (dans_fente_haut || dans_fente_bas)
        return 0.0;

    // Partie solide du mur
    return V0;
    //return 0.5*(x*x+y*y);
   // return 0.0;
}

double W(double x, double y) {
    const double d = 2.0;    // Épaisseur de la couche absorbante (10% de Lx)
    const double W0 = 60.0;  // Intensité de l'absorption

    const double x_lim = param.Lx / 2.0 - d;
    const double y_lim = param.Ly / 2.0 - d;

    double Wx = 0.0;
    double Wy = 0.0;

    if (std::abs(x) > x_lim) {
        double ratio_x = (std::abs(x) - x_lim) / d;
        Wx = W0 * ratio_x * ratio_x * ratio_x; // Profil cubique doux
    }

    if (std::abs(y) > y_lim) {
        double ratio_y = (std::abs(y) - y_lim) / d;
        Wy = W0 * ratio_y * ratio_y * ratio_y;
    }

    return Wx + Wy;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// INITIAL CONDITION /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

std::complex<double> circulaire(double x, double y, double sigma, double k0x, double k0y) {
    double enveloppe = std::exp(-((x - param.x_centre) * (x - param.x_centre)+(y-param.y_centre)*(y-param.y_centre)) / (2.0 * sigma * sigma));
    std::complex<double> phasex = std::exp(1i * k0x * x);
    std::complex<double> phasey = std::exp(1i * k0y * y);    
    return enveloppe * phasex* phasey;
}
///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// PLANE //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

std::complex<double> plane(double x, double y, double sigma, double k0x, double k0y) {
    double enveloppe = std::exp(-((x - param.x_centre) * (x - param.x_centre)) / (2.0 * sigma * sigma));
    std::complex<double> phasex = std::exp(1i * k0x * x);
    return enveloppe * phasex;
}

std::complex<double> condition_initiale(double x, double y, double sigma, double k0x, double k0y) {
    return plane(x, y, sigma, k0x, k0y);
}

std::complex<double> Potentiel_Inhomogene(double x, double y, double sigma, double k0x, double k0y, double t_phys) {
    double enveloppe = std::exp(-((x - param.x_centre) * (x - param.x_centre)) / (2.0 * sigma * sigma));
    std::complex<double> phasex = std::exp(1i * k0x * x);
    double omega = 0.5 * (k0x * k0x + k0y * k0y);

    return enveloppe * phasex * std::exp(-1i * omega * t_phys);
}



#endif /*POTENTIEL_HPP*/