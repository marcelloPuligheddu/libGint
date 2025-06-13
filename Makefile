# Tools
HIPCC = hipcc
CLANGPP = amdclang++

# Flags
COMMON_FLAGS = -std=c++17 -O3 -fPIC -I/opt/rocm-6.4.0/include -I/opt/rocm-6.4.0/include/hipblas

HIP_FLAGS = $(COMMON_FLAGS) --offload-arch=gfx942 # -fopenmp
OMP_GPU_FLAGS = $(COMMON_FLAGS) -fopenmp --offload-arch=gfx942 -I/opt/rocm-6.4.0/include -I/opt/rocm-6.4.0/include/hipblas -D__LIBGINT_OMP_OFFLOAD

# Directories
SRC_DIR = src
BUILD_DIR = build
LIB_DIR = lib

# Files
SPECIAL_CPP = $(SRC_DIR)/prepare_Fm_omp.cpp
SPECIAL_OBJ = $(BUILD_DIR)/prepare_Fm_omp.o
SPECIAL_A = $(LIB_DIR)/prepare_Fm_omp.a

CPP_FILES := $(filter-out $(SPECIAL_CPP), $(wildcard $(SRC_DIR)/*.cpp))
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_FILES))
A_FILES := $(patsubst $(BUILD_DIR)/%.o, $(LIB_DIR)/lib%.a, $(OBJ_FILES))

FINAL_LIB = $(LIB_DIR)/libfinal.a

# Default target
all: $(FINAL_LIB)

# Final static library (combine all .a)
$(FINAL_LIB): $(A_FILES) $(SPECIAL_A)
	@echo "Creating final static library: $@"
	@rm -f $@
	@printf 'CREATE $@\n' > ar_script.mri
	@$(foreach a,$(A_FILES), printf 'ADDLIB %s\n' $(a) >> ar_script.mri;)
	@printf 'ADDLIB %s\nSAVE\nEND\n' $(SPECIAL_A) >> ar_script.mri
	@ar -M < ar_script.mri
	@rm -f ar_script.mri

# Compile HIP/CPU-side OMP files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(HIPCC) $(HIP_FLAGS) -c $< -o $@

# Compile special.cpp with OpenMP GPU offload
$(SPECIAL_OBJ): $(SPECIAL_CPP) | $(BUILD_DIR)
	$(CLANGPP) $(OMP_GPU_FLAGS) -c $< -o $@

# Archive each .o into a static lib
$(LIB_DIR)/lib%.a: $(BUILD_DIR)/%.o | $(LIB_DIR)
	ar rcsD $@ $<

$(SPECIAL_A): $(SPECIAL_OBJ) | $(LIB_DIR)
	ar rcsD $@ $<


# Directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

.PHONY: all clean
