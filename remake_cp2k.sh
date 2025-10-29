set -x
#CP2K_ROOT="${1:-/home/aac/shared/teams/dcgpu_training/epcc/marcellop/fork_cp2k_exp}"
CP2K_ROOT=/home/ubuntu/cp2k/
make install -j 8 PREFIX=${CP2K_ROOT}/tools/toolchain/install/libGint-EXP

if [ $? -eq 0 ]; then

	cd ${CP2K_ROOT}

	rm -f exe/local_cuda/cp2k.psmp
	make -j 1 ARCH=local_cuda VERSION="psmp"
	wait

	cd -
else
	echo " libGint compilation failed"
fi
set +x



