CXX := mpicxx
TARGET := a.out
SRC := $(wildcard *.cpp)

BREW_PREFIX := $(shell brew --prefix)
FFTW_PREFIX := $(shell brew --prefix fftw)

CXXFLAGS := -std=c++20 -O3 \
            -I$(BREW_PREFIX)/include \
            -I$(BREW_PREFIX)/include/eigen3 \
            -I$(FFTW_PREFIX)/include

LDFLAGS := -L$(BREW_PREFIX)/lib -L$(FFTW_PREFIX)/lib
LDLIBS := -lfftw3_mpi -lfftw3 -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) $(LDLIBS) -o $(TARGET)

run: $(TARGET)
	mpirun ./$(TARGET)

clean:
	rm -f $(TARGET)
