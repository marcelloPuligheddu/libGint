
# LibGint

A library for the calculation of the Hartree Fock exchange on GPUs

# Integration into CP2K

The development version of the cp2k wrapper is at https://github.com/marcelloPuligheddu/cp2k

It includes changes to the cp2k make system to automatically download and install libGint

Compared to a regular XF calcalation, the changes needed in the input are :

&FORCE_EVAL  &DFT    &XC      &HF        HFX_LIBRARY libGint

&FORCE_EVAL  &DFT    &XC      &HF        &MEMORY          MAX_MEMORY X

Where X is the maximum amount (in MB) of device memory each MPI rank is able to use during the exchange calculation.

E.g.
```
    &XC
      &HF
        HFX_LIBRARY libGint
        &INTERACTION_POTENTIAL
          CUTOFF_RADIUS 6
          POTENTIAL_TYPE TRUNCATED
          T_C_G_DATA ./t_c_g.dat
        &END INTERACTION_POTENTIAL
        &MEMORY
          MAX_MEMORY 7000
        &END MEMORY
        &SCREENING
          EPS_SCHWARZ 1.0E-11
        &END SCREENING
      &END HF
    &END XC
```

# Limitations

Enough device memory must be available for libGint to keep the density matrix and the Fock matrix on device, one copy per MPI rank. OpenMP threads belonging to the same MPI rank share this memory

Max angular moment: 3 ( f orbitals )

Max number of periodic cells: 256

Max number of linear combinations for each angular moment per set: 4

Max number of primitives gaussian per set: 16 


