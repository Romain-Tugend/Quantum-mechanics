#ifndef POTENTIEL_HPP
#define POTENTIEL_HPP

#include "Common.hpp"
#include "Parametres.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// POTENTIAL ENERGY FUNCTION ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

double V(double x, double y, double W) {
    static std::vector<double> grille_V;
    if (grille_V.empty()) {
        grille_V.assign(param.Nx * param.Ny, 0.0);
        if (W > 0.0) {
            std::vector<double> bruit(param.Nx * param.Ny);
            std::mt19937 gen(42);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (double& b : bruit) b = dist(gen);

            double max_val = 0.0;
            for (int n = 0; n < param.Nx; ++n) {
                for (int m = 0; m < param.Ny; ++m) {
                    double s = 0.0, p = 0.0;
                    for (int dn = -2; dn <= 2; ++dn) {
                        for (int dm = -2; dm <= 2; ++dm) {
                            double w = std::exp(-(dn * dn + dm * dm) / 2.0);
                            s += w * bruit[((n + dn + param.Nx) % param.Nx) * param.Ny + ((m + dm + param.Ny) % param.Ny)];
                            p += w;
                        }
                    }
                    grille_V[n * param.Ny + m] = s / p;
                    max_val = std::max(max_val, std::abs(grille_V[n * param.Ny + m]));
                }
            }
            for (double& val : grille_V) val = W * (val / max_val);
        }
    }
    int n = std::clamp(static_cast<int>(std::round((x + param.Lx / 2.0) / param.dx)), 0, param.Nx - 1);
    int m = std::clamp(static_cast<int>(std::round((y + param.Ly / 2.0) / param.dy)), 0, param.Ny - 1);
    return grille_V[n * param.Ny + m];
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