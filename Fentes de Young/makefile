CXX := mpicxx
TARGET := schrodinger
SRC := $(wildcard *.cpp)

# Détection automatique de Homebrew (macOS) vs système standard (Linux)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    BREW_PREFIX := $(shell brew --prefix)
    FFTW_PREFIX := $(shell brew --prefix fftw)
    INC := -Iinclude -I$(BREW_PREFIX)/include -I$(FFTW_PREFIX)/include
    LIB := -L$(BREW_PREFIX)/lib -L$(FFTW_PREFIX)/lib
else
    INC := -Iinclude
    LIB :=
endif

CXXFLAGS := -std=c++20 -O3 -Wall $(INC)
LDLIBS := $(LIB) -lfftw3_mpi -lfftw3 -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(LDLIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)
