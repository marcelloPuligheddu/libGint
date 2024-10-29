set -x

CP2K_ROOT=$1 # "/home/ubuntu/cp2k_libgint_integration_project/cp2k/"

make install -j 8 PREFIX=${CP2K_ROOT}/tools/toolchain/install/libGint-EXP
wait

cd ${CP2K_ROOT}

rm -f exe/local_cuda/cp2k.psmp
make -j 16 ARCH=local_cuda VERSION="psmp"
wait

cd -

set +x



