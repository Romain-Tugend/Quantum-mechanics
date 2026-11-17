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
    double T=30; // TOTAL TIME OF THE OBSERVATION
    double dx=Lx/Nx; // SPATIAL STEP ON THE X-AXIS
    double dy=Ly/Ny; // SPATIAL STEP ON THE Y-AXIS
    double dt=1e-3; // TEMPORAL STEP 
    int Nt = static_cast<int>(T/dt); // NUMBER OF TEMPORAL POINTS
    int nombre_images=T*1.5*24; // FRAMES PER SECONDES 
    int pas_enregistrement=Nt/nombre_images; // RECORDING FREQUANCY
    double k0x = 0.; // impulsion initiale vers la droite en +
    double k0y = 0.; // impulsion initiale vers la droite en +
    double sigma = 3.0; // ECARTEMENT 
    double x_centre = -0.0; // -gauche; + droite
    double y_centre = -0.0;
    double champ_E_x=0.; //VALUE OF ELETRIC FIELD FOR BLOCH OSSILATIONS X-AXIS
    double champ_E_y=0.; //VALUE OF ELETRIC FIELD FOR BLOCH OSSILATIONS Y-AXIS
    double a_x =1.5; // LATTICE CONSTANTE FOR X-AXIS 
    double a_y =2.; // LATTICE CONSTANTE FOR Y-AXIS
};

#endif /* PARAMETRES_HPP */