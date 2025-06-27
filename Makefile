CXX = g++-12
FC = gfortran
AR = ar
ARFLAGS = rcs

CXXFLAGS = -c -fopenmp -foffload=nvptx-none -fcf-protection=none -no-pie -D__LIBGINT_OMP_OFFLOAD -std=c++17
FCFLAGS = -c
LIBDIR = lib
SRCDIR = src
OBJDIR = obj
TARGET = $(LIBDIR)/libcp2kGint.a

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

# Clean up
clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
