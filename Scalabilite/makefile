CXX := mpicxx
TARGET := a.out
SRC := main.cpp fftw-mpi.cpp

CXXFLAGS := -std=c++20 -O3 -I../../libs/fftw-linux/include
LDFLAGS := -L../../libs/fftw-linux/lib
LDLIBS := -lfftw3_mpi -lfftw3 -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) $(LDLIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)
