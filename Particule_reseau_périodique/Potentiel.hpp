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



#endif /*POTENTIEL_HPP*/