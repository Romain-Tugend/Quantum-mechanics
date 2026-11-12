#ifndef PARAMETRES_HPP
#define PARAMETRES_HPP

#include "Common.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// PARAMETRES //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

struct Parametres
{
    int Nx=256; // NUMBERS OF POINTS ON THE X-AXIS
    int Ny=256; // NUMBERS OF POINTS ON THE Y-AXIS
    double Lx=20.0; // SIZE OF THE DOMAIN FOR THE X-AXIS
    double Ly=20.0; // SIZE OF THE DOMAIN FOR THE Y-AXIS
    double T=5; // TOTAL TIME OF THE OBSERVATION
    double dx=Lx/Nx; // SPATIAL STEP ON THE X-AXIS
    double dy=Ly/Ny; // SPATIAL STEP ON THE Y-AXIS
    double dt=1e-3; // TEMPORAL STEP 
    int Nt = static_cast<int>(T/dt); // NUMBER OF TEMPORAL POINTS
    int nombre_images=T*1.5*24; // FRAMES PER SECONDES 
    int pas_enregistrement=Nt/nombre_images; // RECORDING FREQUANCY
    double k0x = 8.0; // impulsion initiale vers la droite en +
    double k0y = 0.0; // impulsion initiale vers la droite en +
    double sigma = 0.8; // ECARTEMENT 
    double x_centre = -5.0; // -gauche; + droite
    double y_centre = -0.0;
};

#endif /* PARAMETRES_HPP */