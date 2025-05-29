# Compiler and Options
NVCC = hipcc

GFORTRAN = gfortran
PREFIX ?= $(shell pwd)
NVCC_C_OPTS = -v -fPIC -L $(shell pwd)/obj -I $(shell pwd)/src --offload-arch=gfx942 -std=c++17 -I'/opt/rocm-6.4.0/include' -I'/opt/rocm-6.4.0/include/hipblas' 


GFORTRAN_OPTS = -O3 -g -fopenmp
LDFLAGS=-L $(shell pwd)/obj

LIS = -L $(shell pwd)/obj 
# Directories
SRC_DIR = src
OBJ_DIR = obj

# Objects. libGint.o is a bit special

OBJECTS_2 =  $(OBJ_DIR)/interface_libgint.o 
OBJECTS_3 = $(OBJ_DIR)/extern_functions.a $(OBJ_DIR)/plan.a $(OBJ_DIR)/UniqueArray.a  $(OBJ_DIR)/libGint.a $(OBJ_DIR)/util.a $(OBJ_DIR)/fgamma.a $(OBJ_DIR)/compute_Fm.a $(OBJ_DIR)/compute_VRR.a $(OBJ_DIR)/compute_VRR2.a $(OBJ_DIR)/compute_ECO.a $(OBJ_DIR)/compute_HRR.a $(OBJ_DIR)/compute_SPH.a $(OBJ_DIR)/compute_TRA.a $(OBJ_DIR)/compute_KS.a $(OBJ_DIR)/t_c_g0_n.a

# Targets
all: pre_install libcp2kGint.a

# Compilation rules
$(OBJ_DIR)/%.a: $(SRC_DIR)/%.cpp
	$(NVCC) --emit-static-lib $(NVCC_C_OPTS) $(LDFLAGS) $< -o $@ $(LIS)

$(OBJ_DIR)/interface_libgint.o: $(SRC_DIR)/interface_libgint.F90
	$(GFORTRAN) $(GFORTRAN_OPTS) -c $< -o $@ -lstdc++

# Static library creation
libcp2kGint.a: $(OBJECTS_3)
	ar rcsD $@ $^`

pre_install:
	mkdir -p obj/

# File copying and packaging
install: all
	mkdir -p $(PREFIX)
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	
	cp libcp2kGint.a $(PREFIX)/lib
	cp libgint.mod $(PREFIX)/include
	
# Clean the build
clean:
	rm -f $(OBJ_DIR)/*.a libcp2kGint.a *.mod

.PHONY: all clean install pre_install
