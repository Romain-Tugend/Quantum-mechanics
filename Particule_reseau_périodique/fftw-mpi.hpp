#ifndef FFTW_MPI_HPP
#define FFTW_MPI_HPP
#include "Common.hpp"
#include "fftw-mpi.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// FAST FOURIER TRANSFORM WITH fftw3 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

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

    FFT2D(int Nx, int Ny): size_total_x(Nx), size_total_y(Ny){
        alloc_local = fftw_mpi_local_size_2d(size_total_x, size_total_y, MPI_COMM_WORLD, &local_n0, &local_0_start);
        in = fftw_alloc_complex(alloc_local);
        out = fftw_alloc_complex(alloc_local);
        pfft = fftw_mpi_plan_dft_2d(size_total_x, size_total_y, in, out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
        pifft = fftw_mpi_plan_dft_2d(size_total_x, size_total_y, in, out, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    // Calcule la FFT directe (équivalent à np.fft.fft)
    void appliquer_fft(const std::vector<std::complex<double>>& input, std::vector<std::complex<double>>& output){
        std::complex<double>* in_ptr = reinterpret_cast<std::complex<double>*>(in);
        for (int i = 0; i < local_n0; ++i) {
            for (int j = 0; j< size_total_y; j++){
                in_ptr[i*size_total_y+j]= input[(i)*size_total_y+j];
            }
        }
        fftw_execute(pfft);
        std::complex<double>* out_ptr = reinterpret_cast<std::complex<double>*>(out);
        for (int i = 0; i < local_n0; ++i) {
            for (int j = 0; j< size_total_y; j++){
                output[(i)*size_total_y+j]=out_ptr[i*size_total_y+j];
            }
        }
    }
    // Calcule la FFT inverse (équivalent à np.fft.ifft)
    void appliquer_ifft(const std::vector<std::complex<double>>& input, std::vector<std::complex<double>>& output) {
        std::complex<double>* in_ptr = reinterpret_cast<std::complex<double>*>(in);
        for (int i = 0; i < local_n0; ++i) {
            for (int j=0; j<size_total_y; j++){
                in_ptr[i*size_total_y+j]= input[(i)*size_total_y+j];
            }
        }
        fftw_execute(pifft);
        double norm = static_cast<double>(size_total_x * size_total_y);
        std::complex<double>* out_ptr = reinterpret_cast<std::complex<double>*>(out);
        for (int i = 0; i < local_n0; ++i) {
            for (int j = 0; j< size_total_y; j++){
                output[(i)*size_total_y+j]=out_ptr[i*size_total_y+j];
                output[(i)*size_total_y+j] /= norm;
            }
        }
    }
    ~FFT2D(){
        fftw_destroy_plan(pifft);
        fftw_destroy_plan(pfft);
        fftw_free(in);
        fftw_free(out);
    }
};

#endif /* FFTW_MPI_HPP */