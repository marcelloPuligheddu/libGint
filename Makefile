# Compiler and Options
NVCC = hipcc
MPICPP = mpic++
GFORTRAN = gfortran

ARCH_NUM ?= 70
PREFIX ?= $(shell pwd)
NVCC_C_OPTS =  -std=c++17 -I'/opt/rocm-6.4.0/include' -I'/opt/rocm-6.4.0/include/hipblas' -fgpu-rdc 
NVCC_D_OPTS =  -lgomp -L'/opt/rocm-6.4.0/lib' -v
MPICPP_EX_OPTS = -std=c++17 -Wall -fPIC -O3
GFORTRAN_OPTS = -O3 -g -fopenmp

LIS = 
# Directories
SRC_DIR = src
OBJ_DIR = obj

# Objects. libGint.o is a bit special
OBJECTS_1 = 
# $(OBJ_DIR)/libGint.o
OBJECTS_2 =  $(OBJ_DIR)/interface_libgint.o $(OBJ_DIR)/extern_functions.o $(OBJ_DIR)/plan.o $(OBJ_DIR)/UniqueArray.o  $(OBJ_DIR)/libGint.o $(OBJ_DIR)/util.o $(OBJ_DIR)/fgamma.o $(OBJ_DIR)/compute_Fm.o $(OBJ_DIR)/compute_VRR.o $(OBJ_DIR)/compute_VRR2.o $(OBJ_DIR)/compute_ECO.o $(OBJ_DIR)/compute_HRR.o $(OBJ_DIR)/compute_SPH.o $(OBJ_DIR)/compute_TRA.o $(OBJ_DIR)/compute_KS.o $(OBJ_DIR)/t_c_g0_n.o

# Targets
all: pre_install libcp2kGint.a

# Compilation rules
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) -c $(NVCC_C_OPTS) $< -o $@ $(LIS)

#$(OBJ_DIR)/libGint_unlinked.o: $(SRC_DIR)/libGint.cpp
#	$(NVCC) -c $(NVCC_C_OPTS) $< -o $@ $(LIS)

#$(OBJ_DIR)/libGint.o: $(OBJ_DIR)/libGint_unlinked.o # $(OBJECTS_2)
#	$(NVCC) -c $(NVCC_C_OPTS) -o $@ $^ $(LIS)


$(OBJ_DIR)/interface_libgint.o: $(SRC_DIR)/interface_libgint.F90
	$(GFORTRAN) $(GFORTRAN_OPTS) -c $< -o $@ -lstdc++ -lcudadevrt -lcudart 

# Static library creation
libcp2kGint.a: $(OBJECTS_1) $(OBJECTS_2)
	ar -rcs $@ $^

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
	rm -f $(OBJ_DIR)/*.o libcp2kGint.a *.mod

.PHONY: all clean install pre_install
