#ifndef FFTW_MPI_HPP
#define FFTW_MPI_HPP
#include "Common.hpp"

class FFT2D
{
    private:
    fftw_complex *in;
    fftw_complex *out;
    fftw_plan pfft;
    fftw_plan pifft;
    int size_total_x;
    int size_total_y;

    public:
    FFT2D(const FFT2D&) = delete;             
    FFT2D& operator=(const FFT2D&) = delete;  
    ptrdiff_t alloc_local, local_n0, local_0_start; 

    FFT2D(int Nx, int Ny);
    // Calcule la FFT directe (équivalent à np.fft.fft)
    void appliquer_fft(const std::vector<std::complex<double>>& input,
        std::vector<std::complex<double>>& output
    );
    
    // Calcule la FFT inverse (équivalent à np.fft.ifft)
    void appliquer_ifft(const std::vector<std::complex<double>>& input, 
        std::vector<std::complex<double>>& output
    );
    ~FFT2D();
};

#endif /* FFTW_MPI_HPP */