#ifndef POTENTIEL_HPP
#define POTENTIEL_HPP

#include "Common.hpp"
#include "Parametres.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// POTENTIAL ENERGY FUNCTION ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

double V(double x, double y, double champ_E_x, double champ_E_y, double a_x, double a_y) {
    const double V0 = 4.0;       // Hauteur des barrières de potentiel
    const double k_reseau_x = M_PI / a_x; 
    const double k_reseau_y = M_PI / a_y;
    double barriere_x = std::sin(k_reseau_x * x);
    double barriere_y = std::sin(k_reseau_y * y);
    double potentiel_periodique = V0 * (barriere_x * barriere_x + barriere_y * barriere_y);
    
    double potentiel_electro=-champ_E_x*x-champ_E_y*y;

    return potentiel_periodique+potentiel_electro;
    
}

double V_hex(double x, double y, double Ex, double Ey) {
    const double V0 = 15.0;
    const double k = 2.0 * M_PI / 2.5;
    double u1 = k * x;
    double u2 = k * (-0.5 * x + 0.8660254 * y);
    double u3 = k * (-0.5 * x - 0.8660254 * y);
    double reseau = V0 * (std::cos(u1) + std::cos(u2) + std::cos(u3));
    return reseau - Ex * x - Ey * y;
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



#endif /*POTENTIEL_HPP*/