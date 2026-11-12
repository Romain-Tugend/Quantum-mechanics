#ifndef POTENTIEL_HPP
#define POTENTIEL_HPP

#include "Common.hpp"
#include "Parametres.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// POTENTIAL ENERGY FUNCTION ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

double V(double x, double y) {
    const double V0 = 50.0;       // Hauteur des barrières de potentiel
    const double a = 1.0;         // Pas du réseau (distance entre deux "atomes")
    
    const double k_reseau = M_PI / a; 

    double barriere_x = std::sin(k_reseau * x);
    double barriere_y = std::sin(k_reseau * y);
    
    double potentiel_periodique = V0 * (barriere_x * barriere_x + barriere_y * barriere_y);

    return potentiel_periodique;
    
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