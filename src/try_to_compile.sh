set -x

accellerator=
#CPPFLAGS="-mp=$accellerator -Minfo=accel -O2 -pg -lblas"
#CPP=nvc++
CPPFLAGS="-Ofast -march=native -g -pg -lblas -Wall -pedantic -lcudart "
CPP=nvc++

rm compute_Fm.o compute_VRR.o compute_HRR.o compute_TRA.o compute_SPH.o 
rm util.o plan.o AIS.o fgamma.o
rm test_libGint

$CPP compute_Fm.cpp  -c -o compute_Fm.o  $CPPFLAGS -x cu -cuda &&
$CPP compute_VRR.cpp -c -o compute_VRR.o $CPPFLAGS -x cu -cuda &&
$CPP compute_HRR.cpp -c -o compute_HRR.o $CPPFLAGS -x cu -cuda &&
$CPP compute_TRA.cpp -c -o compute_TRA.o $CPPFLAGS -x cu -cuda &&
$CPP compute_SPH.cpp -c -o compute_SPH.o $CPPFLAGS -x cu -cuda &&
$CPP fgamma.cpp -c -o fgamma.o $CPPFLAGS -x cu -cuda &&
$CPP UniqueArray.cpp -c -o UniqueArray.o $CPPFLAGS &&
$CPP util.cpp -c -o util.o               $CPPFLAGS -x cu &&
$CPP plan.cpp -c -o plan.o               $CPPFLAGS &&
$CPP AIS.cpp  -c -o AIS.o                $CPPFLAGS -x cu -cuda &&
$CPP main.cpp \
   plan.o util.o fgamma.o AIS.o UniqueArray.o \
   compute_Fm.o compute_VRR.o compute_HRR.o compute_TRA.o compute_SPH.o \
   -o test_libGint \
   $CPPFLAGS -cuda -lcublas

set +x
