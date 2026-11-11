#ifndef COMMON_HPP
#define COMMON_HPP

#include<iostream>
#include<cmath>
#include<vector>
#include<fftw3-mpi.h>
#include<complex>
#include <chrono> 
#include <fstream>
#include <mpi.h>
#include "fftw-mpi.hpp"
#include "Parametres.hpp"

using namespace std;
using namespace std::literals::complex_literals;

inline Parametres param;

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// GLOBAL VECTORS ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
inline std::vector<double>lx(param.Nx);
inline std::vector<double>ly(param.Ny);
inline std::vector<double>k_physique_x(param.Nx);
inline std::vector<double>k_physique_y(param.Ny);
inline std::vector<std::complex<double>>Potentiel_V_global(param.Nx*param.Ny);
inline std::vector<std::complex<double>>Matrice_k_global(param.Nx*param.Ny);
inline std::vector<std::complex<double>>lf_global(param.Nx * param.Ny);

///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// MPI VECTS //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
inline std::vector<int> counts;
inline std::vector<int> displs;

#endif /* COMMON_HPP */