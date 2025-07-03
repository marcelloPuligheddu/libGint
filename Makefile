PREFIX ?= $(shell pwd)

CXX = g++-12
FC = gfortran-12
AR = ar
ARFLAGS = rcs 


CXXFLAGS = -c -fopenmp -foffload=nvptx-none -fcf-protection=none -no-pie -D__LIBGINT_OMP_OFFLOAD -std=c++17 -save-temps -save-temps=obj -fno-stack-protector
FCFLAGS = -c
LIBDIR = lib
SRCDIR = src
OBJDIR = obj
TARGET = $(LIBDIR)/libcp2kGint.a

# FCFLAGS = -foffload=nvptx-none -fopenmp test_libGint.o -Wl,--whole-archive lib/libcp2kGint.a -Wl,--no-whole-archive -o test_libGint -lc -lstdc++

CPP_SRCS = $(wildcard $(SRCDIR)/*.cpp)
CPP_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SRCS))

F90_SRC = $(SRCDIR)/interface_libgint.F90
F90_OBJ = $(OBJDIR)/interface_libgint.o

OBJS = $(CPP_OBJS) $(F90_OBJ)

# Default target
all: $(TARGET)

# Create the static library
$(TARGET): $(OBJS) | $(LIBDIR)
	$(AR) $(ARFLAGS) $@ $^

# Compile C++ sources
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# Compile Fortran source
$(OBJDIR)/interface_libgint.o: $(F90_SRC) | $(OBJDIR)
	$(FC) $(FCFLAGS) $< -o $@

# Create directories if they don't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

install: all
	mkdir -p $(PREFIX)
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	
	cp $(TARGET) $(PREFIX)/lib
	cp libgint.mod $(PREFIX)/include

test: $(TARGET)
	$(FC) -c -fopenmp -c test_libGint.f90 -o test_libGint.o
	$(FC) -foffload=nvptx-none -fopenmp test_libGint.o -Wl,--whole-archive lib/libcp2kGint.a -Wl,--no-whole-archive -o test_libGint -lc -lstdc++ -foffload-options=nvptx-none="-lm" -no-pie -save-temps 
	./test_libGint

# Clean up
clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
