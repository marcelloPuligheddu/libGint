# amdclang++ -fopenmp --offload-arch=gfx942 -c -static src/prepare_Fm_omp.cpp -o obj/prepare_Fm_omp.a -D__LIBGINT_OMP_OFFLOAD

MAKE_HIP_CPP="hipcc -x hip --emit-static-lib -g -fPIC -fPIE --offload-arch=gfx942 --std=c++17 -I /opt/rocm-6.4.0/include  -I /opt/rocm-6.4.0/include/hipblas "
${MAKE_HIP_CPP} src/util.cpp -o obj/util.a
${MAKE_HIP_CPP} src/UniqueArray.cpp -o obj/UniqueArray.a
${MAKE_HIP_CPP} src/t_c_g0_n.cpp -o obj/t_c_g0_n.a
${MAKE_HIP_CPP} src/compute_Fm.cpp -o src/compute_Fm.a -l obj/UniqueArray.a -l ./obj/util.a -l /obj/t_c_g0_n.a



