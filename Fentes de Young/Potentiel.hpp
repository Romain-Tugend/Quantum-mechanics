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



#endif /*POTENTIEL_HPP*/