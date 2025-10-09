
!> @brief Handles initialization and interface with the libGint library.
!>
!> This module provides Fortran wrappers for the libGint C++ backend,
!> including creation of libGint handles, environment setup, and
!> Fock matrix updates.
!>
!> @details
!> The functions in this module act as a bridge between Fortran
!> and the C++ implementation of libGint. They use opaque C pointers
!> (c_ptr) to store handles to C++ objects.
!>
!> Interfaces are not commented, see the subroutines later on for documentation
!>
!> @author M Puligheddu
!> 
!> @date 2025-10-09

module libgint
   use, intrinsic :: iso_c_binding, only : c_loc, c_ptr, c_null_ptr, c_int, c_bool
   implicit none
   private
   interface
   function libgint_internal_create_handle() result ( libgint_handle ) bind(C,name="libgint_create_handle")
      import :: c_ptr
      implicit none
      type(c_ptr) :: libgint_handle 
   end function libgint_internal_create_handle
   end interface

   interface
   subroutine libgint_internal_init( libgint_handle ) bind(C,name="libgint_init")
      import :: c_ptr
      implicit none
      type(c_ptr), value :: libgint_handle
   end subroutine libgint_internal_init
   end interface

   interface
   subroutine libgint_internal_set_hf_fac( libgint_handle, fac ) bind(C,name="libgint_set_hf_fac")
      import :: c_ptr
      implicit none
      real(kind=8), value :: fac
      type(c_ptr), value :: libgint_handle
   end subroutine libgint_internal_set_hf_fac
   end interface

   interface
   subroutine libgint_internal_set_max_mem( libgint_handle, max_mem ) bind(C,name="libgint_set_max_mem")
      import :: c_ptr, c_int
      implicit none
      integer(kind=c_int), value :: max_mem
      type(c_ptr), value :: libgint_handle
   end subroutine libgint_internal_set_max_mem
   end interface

   interface
   subroutine libgint_internal_set_P( libgint_handle, P, P_size ) bind(C,name="libgint_set_P")
      import :: c_ptr, c_int
      implicit none
      type(c_ptr), value :: libgint_handle, P
      integer(kind=c_int), value :: P_size
   end subroutine libgint_internal_set_P
   end interface

   interface
   subroutine libgint_internal_set_P_polarized( libgint_handle, Pa, Pb, P_size ) bind(C,name="libgint_set_P_polarized")
      import :: c_ptr, c_int
      implicit none
      type(c_ptr), value :: libgint_handle, Pa, Pb
      integer(kind=c_int), value :: P_size
   end subroutine libgint_internal_set_P_polarized
   end interface

   interface
   subroutine libgint_internal_set_K( libgint_handle, K, K_size, fac ) bind(C,name="libgint_set_K")
      import :: c_ptr, c_int
      implicit none
      type(c_ptr), value :: libgint_handle, K
      integer(kind=c_int), value :: K_size
      real(kind=8), value :: fac
   end subroutine libgint_internal_set_K
   end interface

   interface
   subroutine libgint_internal_set_K_polarized( libgint_handle, Ka, Kb, K_size ) bind(C,name="libgint_set_K_polarized")
      import :: c_ptr, c_int
      implicit none
      type(c_ptr), value :: libgint_handle, Ka, Kb
      integer(kind=c_int), value :: K_size
   end subroutine libgint_internal_set_K_polarized
   end interface

   interface
   subroutine libgint_internal_get_K( libgint_handle, K ) bind(C,name="libgint_get_K")
      import :: c_ptr, c_int
      implicit none
      type(c_ptr), value :: libgint_handle, K
   end subroutine libgint_internal_get_K
   end interface

   interface
   subroutine libgint_internal_get_K_polarized( libgint_handle, Ka, Kb ) bind(C,name="libgint_get_K_polarized")
      import :: c_ptr, c_int
      implicit none
      type(c_ptr), value :: libgint_handle, Ka, Kb
   end subroutine libgint_internal_get_K_polarized
   end interface

   interface 
   subroutine libgint_internal_set_Atom( libgint_handle, i, R, Z, np ) bind(C,name="libgint_set_Atom")
      import :: c_ptr, c_int
      implicit none
      integer(kind=c_int), value :: i, np
      type (c_ptr), value :: libgint_handle, R ,Z
   end subroutine libgint_internal_set_Atom
   end interface

   interface 
   subroutine libgint_internal_set_Atom_L( libgint_handle, i, l, nl, K ) bind(C,name="libgint_set_Atom_L")
      import :: c_ptr, c_int
      implicit none
      integer(kind=c_int), value :: i, l, nl
      type (c_ptr), value :: libgint_handle, K
   end subroutine libgint_internal_set_Atom_L
   end interface

   interface
   subroutine libgint_internal_set_AtomInfo( libgint_handle, i, R, Z, np, lmin, Lmax, nl, K ) bind(C,name="libgint_set_AtomInfo")
      import c_ptr, c_int
      implicit none
      integer(kind=c_int), value :: i, np, lmin, Lmax
      type(c_ptr), value :: libgint_handle, R,Z,nl,K
   end subroutine libgint_internal_set_AtomInfo
   end interface

   interface
   subroutine libgint_internal_set_cell( libgint_handle, periodic, cell_h, cell_i ) bind(C,name="libgint_set_cell")
      import :: c_ptr, c_int, c_bool
      implicit none
      logical(kind=c_bool) :: periodic
      type(c_ptr), value :: libgint_handle, cell_h,cell_i
   end subroutine libgint_internal_set_cell
   end interface

! void libgint_set_neighs( void * handle, double * neighs_, int nneighs ){
   interface
   subroutine libgint_internal_set_neighs( libgint_handle, neighs, nneighs ) bind(C,name="libgint_set_neighs")
      import :: c_ptr, c_int
      implicit none
      type(c_ptr), value :: libgint_handle, neighs
      integer(kind=c_int), value :: nneighs
   end subroutine libgint_internal_set_neighs
   end interface

   interface
   subroutine libgint_internal_add_prm( libgint_handle, ipa,ipb,ipc,ipd ) bind (C,name="libgint_add_prm")
      import :: c_ptr, c_int
      implicit none
      integer(kind=c_int), value :: ipa,ipb,ipc,ipd
      type (c_ptr), value :: libgint_handle
   end subroutine libgint_internal_add_prm
   end interface

   interface   
   subroutine libgint_internal_add_shell( libgint_handle, i,j,k,l,n1,n2 ) bind (C,name="libgint_add_shell")
      import :: c_ptr, c_int
      implicit none
      integer(kind=c_int), value :: i,j,k,l,n1,n2
      type (c_ptr), value :: libgint_handle
   end subroutine libgint_internal_add_shell
   end interface

   interface
   subroutine libgint_internal_add_cell( libgint_handle ) bind (C,name="libgint_add_cell")
      import :: c_ptr
      implicit none
      type (c_ptr), value :: libgint_handle
   end subroutine libgint_internal_add_cell
   end interface

   interface
   subroutine libgint_internal_add_qrt( libgint_handle, la,lb,lc,ld,nla,nlb,nlc,nld ) bind (C,name="libgint_add_qrt")
      import :: c_ptr, c_int
      implicit none
      integer(kind=c_int), value :: la,lb,lc,ld,nla,nlb,nlc,nld
      type (c_ptr), value :: libgint_handle
   end subroutine libgint_internal_add_qrt
   end interface

   interface
   subroutine libgint_internal_add_qrtt( libgint_handle, symm_fac, la,lb,lc,ld, inla,inlb,inlc,inld, &
         ld_ac,ld_ad,ld_bc,ld_bd, offset_ac_L_set,offset_ad_L_set, &
         offset_bc_L_set,offset_bd_L_set, Tac,Tad,Tbc,Tbd ) bind (C,name="libgint_add_qrtt")
      import :: c_ptr, c_int, c_bool
      implicit none
      real( kind=8), value :: symm_fac
      integer(kind=c_int), value :: la,lb,lc,ld, inla,inlb,inlc,inld
      integer(kind=c_int), value :: ld_ac,ld_ad,ld_bc,ld_bd
      integer(kind=c_int), value :: offset_ac_L_set, offset_ad_L_set
      integer(kind=c_int), value :: offset_bc_L_set, offset_bd_L_set
      logical(kind=c_bool), value :: Tac,Tad,Tbc,Tbd
      type (c_ptr), value :: libgint_handle
   end subroutine libgint_internal_add_qrtt
   end interface

   interface
   subroutine libgint_internal_add_set( libgint_handle ) bind (C,name="libgint_add_set")
      import :: c_ptr
      implicit none
      type (c_ptr), value :: libgint_handle
   end subroutine libgint_internal_add_set
   end interface

   interface
   subroutine libgint_internal_dispatch ( libgint_handle ) bind (C,name="libgint_dispatch")
      import :: c_ptr
      implicit none
      type (c_ptr), value :: libgint_handle
   end subroutine libgint_internal_dispatch
   end interface

   interface 
   subroutine libgint_internal_set_Potential_Truncated( libgint_handle, R_cut, C0, ld_C0, C0_size ) &
                                                      bind (C,name="libgint_set_Potential_Truncated")
      import :: c_ptr, c_int
      implicit none
      type (c_ptr), value :: libgint_handle, C0
      real(kind=8), value :: R_cut
      integer(kind=c_int), value :: ld_C0, C0_size
   end subroutine libgint_internal_set_Potential_Truncated
   end interface



   public :: libgint_init, libgint_set_Potential_Truncated, libgint_set_hf_fac, libgint_set_max_mem
   public :: libgint_set_P, libgint_set_P_polarized, libgint_set_K, libgint_set_K_polarized
   public :: libgint_get_K, libgint_get_K_polarized, libgint_set_Atom, libgint_set_Atom_L, libgint_set_cell
   public :: libgint_set_neighs
   public :: libgint_add_prm, libgint_add_shell, libgint_add_cell, libgint_add_qrt
   public :: libgint_add_qrtt, libgint_add_set, libgint_dispatch

contains

!> @brief Initializes the libGint engine.
!>
!> This Fortran wrapper allocates and initializes a new libGint context
!> by calling the underlying C++ constructor and initialization routines.
!> The returned handle is an opaque pointer that should be passed
!> to other libGint interface functions.
!>
!> @param[out] handle pointer to a new libGint instance.
   subroutine libgint_init ( handle )
      type (c_ptr) :: handle
      handle = libgint_internal_create_handle()
      call libgint_internal_init( handle )
   end subroutine libgint_init

!> @brief Sets the Truncated Coulomb as the potential, pass R_cut and C0
!>        to libGint internals
   subroutine libgint_set_Potential_Truncated( handle, R_cut, C0 )
      type (c_ptr) :: handle
      real(kind=8) :: R_cut
      integer(kind=c_int) :: ld_C0, C0_size
      real(kind=8), dimension(:,:), target :: C0
      C0_size = size(C0,kind=c_int)
      ld_C0 = size(C0, dim=1, kind=c_int)
      call libgint_internal_set_Potential_Truncated( handle, %val(R_cut), c_loc(C0), %val(ld_C0), %val(C0_size) )
   end subroutine libgint_set_Potential_Truncated

!> @brief Sets the fraction of Hartree Fock exchange, in case of hybrids functionals
!>        to libGint internals
   subroutine libgint_set_hf_fac( handle, fac )
      type(c_ptr) :: handle
      real(kind=8), value :: fac
      call libgint_internal_set_hf_fac( handle, fac )
   end subroutine libgint_set_hf_fac

!> @brief Sets the maximum memory ( in MB ) usable by libGint on gpu per mpi thread
subroutine libgint_set_max_mem( handle, max_mem )
      type(c_ptr) :: handle
      integer(kind=c_int), value :: max_mem
      call libgint_internal_set_max_mem( handle, max_mem )
   end subroutine libgint_set_max_mem

!> @brief Inform libGint of the location of the density matrix, no spin
   subroutine libgint_set_P ( handle, P )
      type (c_ptr) :: handle
      integer(kind=c_int) :: P_size
      real(kind=8), dimension(:), target :: P
      P_size = size(P, kind=c_int )
      call libgint_internal_set_P( handle, c_loc(P), %val(P_size) )
   end subroutine libgint_set_P
   
!> @brief Inform libGint of the location of the density matrix, spin case
   subroutine libgint_set_P_polarized ( handle, Pa, Pb )
      type (c_ptr) :: handle
      integer(kind=c_int) :: P_size
      real(kind=8), dimension(:), target :: Pa, Pb
      P_size = size(Pa, kind=c_int )
      call libgint_internal_set_P_polarized( handle, c_loc(Pa), c_loc(Pb), %val(P_size) )
   end subroutine libgint_set_P_polarized

!> @brief Inform libGint of the location of the fock matrix on cpu.
!> @note  This routine will copy the existing K matrix, only use if 
!>        K is not zero at the start of the HF exchange calculation
!>        If K is zero, do nothing, libGint will manage K internally
!>        until get_K is called
   subroutine libgint_set_K ( handle, K, fac )
      type (c_ptr) :: handle
      integer(kind=c_int) :: K_size
      real(kind=8), dimension(:), target :: K
      real(kind=8), value :: fac
      K_size = size(K, kind=c_int )
      call libgint_internal_set_K( handle, c_loc(K), %val(K_size), fac )
   end subroutine libgint_set_K
   
!> @brief Inform libGint of the location of the fock matrices on cpu.
!>        see libgint_set_K notes for details and use
   subroutine libgint_set_K_polarized ( handle, Ka, Kb )
      type (c_ptr) :: handle
      integer(kind=c_int) :: K_size
      real(kind=8), dimension(:), target :: Ka, Kb
      K_size = size(Ka, kind=c_int )
      call libgint_internal_set_K_polarized( handle, c_loc(Ka), c_loc(Kb), %val(K_size) )
   end subroutine libgint_set_K_polarized

!> @brief when this function returns, K will point to the Fock matrix
   subroutine libgint_get_K ( handle, K )
      type (c_ptr) :: handle
      real(kind=8), dimension(:), target :: K
      call libgint_internal_get_K( handle, c_loc(K) )
   end subroutine libgint_get_K

!> @brief when this function returns, Ks will point to the Fock matrices
   subroutine libgint_get_K_polarized ( handle, Ka, Kb )
      type (c_ptr) :: handle
      real(kind=8), dimension(:), target :: Ka, Kb
      call libgint_internal_get_K_polarized( handle, c_loc(Ka), c_loc(Kb) )
   end subroutine libgint_get_K_polarized

!> @brief sets the simulation box, cell_h is the 3x3 h matrix, cell_i its inverse
!> @param periodic  same meaning as in cp2k. Non periodic may work, but not supported
!> @param cell_h    3x3 H matrix
!> @param cell_h    3x3 H^-1 matrix
   subroutine libgint_set_cell( handle, periodic, cell_h, cell_i )
      type (c_ptr) :: handle
      logical(kind=c_bool), value :: periodic
      real(kind=8), dimension(:,:), target :: cell_h, cell_i
      call libgint_internal_set_cell( handle, periodic, c_loc(cell_h), c_loc(cell_i) )
   end subroutine libgint_set_cell

!> @brief Incredibly badly named, done for consistency with cp2k,
          sets the list of lattice vectors for pbc loops,
          each element is a 3d vector pointing to a box in the lattice
!> @param neighs array with lattice vectors, each element is a 3d vector pointing to a box in the lattice
!> @param nneighs number of lattice vectors
   subroutine libgint_set_neighs( handle, neighs, nneighs )
      type (c_ptr) :: handle
      integer(kind=c_int) :: nneighs
      real(kind=8), dimension(:,:), target :: neighs
!      write (*,*) " Setting neigh as " , neighs
      call libgint_internal_set_neighs( handle, c_loc(neighs), nneighs )
   end subroutine libgint_set_neighs

!> @brief Pass information about set i
!>        Called in combination with libgint_set_Atom_L
!> @param i index of set, i are assumed to increase by one for each new set
!> @param R positon of the atom associated with this set
!> @param Z gaussian coefficients
!> @param np number of gaussian coefficients
   subroutine libgint_set_Atom( handle, i, R, Z, np )
      type (c_ptr) :: handle
      integer(kind=c_int), value :: i, np
      real(kind=8), dimension(:), target :: R, Z
!      write(*,*) " calling set Atom(set) ", i, R, Z, np
      call libgint_internal_set_Atom( handle, i, c_loc(R), c_loc(Z), np )
   end subroutine libgint_set_Atom

!> @brief Pass information about subset/shell of angular moment l of set i
!>        Called in combination with libgint_set_Atom
!> @param i index of set
!> @param l angular moment of this subset/shell
!> @param nl number of linear combinations
!> @param K matrix of contraction coefficients
!> @note R Z and K are populated from libgint_set_Atom_L
   subroutine libgint_set_Atom_L( handle, i, l, nl, K )
      type (c_ptr) :: handle
      integer(kind=c_int), value :: i, l , nl
      real(kind=8), dimension(:), target :: K
!      write(*,*) " calling set Atom L ", i, l, nl , K
      call libgint_internal_set_Atom_L( handle, i, l, nl, c_loc(K) )
   end subroutine libgint_set_Atom_L

!> @brief Unused, see extern_function.cpp for notes
   subroutine ligbint_set_AtomInfo( handle, i, R, Z, np, lmin, Lmax, nl, K )
      type (c_ptr) :: handle
      integer(kind=c_int), value :: i, np, lmin, Lmax
      integer(kind=4), dimension(:), target :: nl
      real(kind=8), dimension(:), target :: R,Z,K
      call libgint_internal_set_AtomInfo( handle, i, c_loc(R), c_loc(Z), np, lmin, Lmax, c_loc(nl) , c_loc(K) )
   end subroutine ligbint_set_AtomInfo

!> @param Add a quartet of primitives to the work list.
!>        called in combination with libgint_add_shell
   subroutine libgint_add_prm( handle, ipa, ipb, ipc, ipd )
      type (c_ptr) :: handle
      integer(kind=c_int), value :: ipa,ipb,ipc,ipd
      call libgint_internal_add_prm( handle, ipa,ipb,ipc,ipd )
   end subroutine libgint_add_prm
!> @brief Add a set to the work list, 
!>        called in combination with libgint_add_prm
!> @note add_prm and add_shell contains enough info to compute the integrals
!>       we still need add_qrt, add_qrtt and add_set to digest the integrals
   subroutine libgint_add_shell( handle, i, j, k, l, n1, n2 )
      type (c_ptr) :: handle
      integer(kind=c_int), value :: i,j,k,l,n1,n2
      call libgint_internal_add_shell( handle, i,j,k,l,n1,n2 )
   end subroutine libgint_add_shell

!> @brief Unused 
   subroutine libgint_add_cell( handle )
      type (c_ptr) :: handle
      call libgint_internal_add_cell( handle )
   end subroutine libgint_add_cell

!> @brief Prepare libGint for the fact that an integral with nlx linear combinations
!>        at angular moments lx will be added to the work list
!>        called in combination with libgint_add_qrtt and libgint_add_set
   subroutine libgint_add_qrt( handle, la,lb,lc,ld, nla,nlb,nlc,nld )
      type (c_ptr) :: handle
      integer(kind=c_int), value :: la,lb,lc,ld, nla,nlb,nlc,nld
      call libgint_internal_add_qrt( handle, la,lb,lc,ld,nla,nlb,nlc,nld )
   end subroutine libgint_add_qrt

!> @brief Add to the work list the digestion of an integral
!>        at angular moments lx, for the linear combination inlx
!>        using a sub-block of the sparse density matrix, of leading dimension ld_xy
!>        pointed to by offset_xy_L_set.
!>        The density matrix is triangular, so we need to know if each sub-block xy
!>        is ordered as x then y or y then x. Txy is true if transposed
!> @note Remeber c and f ordering 
   subroutine libgint_add_qrtt( handle, symm_fac, la,lb,lc,ld, inla,inlb,inlc,inld, &
         ld_ac,ld_ad,ld_bc,ld_bd, offset_ac_L_set,offset_ad_L_set, &
         offset_bc_L_set,offset_bd_L_set, Tac,Tad,Tbc,Tbd )
      type (c_ptr) :: handle
      real( kind=8), value :: symm_fac
      integer(kind=c_int), value :: la,lb,lc,ld, inla,inlb,inlc,inld
      integer(kind=c_int), value :: ld_ac,ld_ad,ld_bc,ld_bd
      integer(kind=c_int), value :: offset_ac_L_set, offset_ad_L_set 
      integer(kind=c_int), value :: offset_bc_L_set, offset_bd_L_set
      logical(kind=c_bool), value :: Tac,Tad,Tbc,Tbd
      call libgint_internal_add_qrtt( handle, symm_fac, la,lb,lc,ld,inla,inlb,inlc,inld, &
         ld_ac,ld_ad,ld_bc,ld_bd, offset_ac_L_set,offset_ad_L_set, &
         offset_bc_L_set,offset_bd_L_set, Tac,Tad,Tbc,Tbd )
   end subroutine libgint_add_qrtt

!> @brief Inform libgint we are done with this particular quartet of sets.
   subroutine libgint_add_set( handle )
      type (c_ptr) :: handle
      call libgint_internal_add_set( handle )
   end subroutine libgint_add_set

!> @brief Force libGint to compute K NOW. Generally better to wait for get_K
subroutine libgint_dispatch( handle )
      type (c_ptr) :: handle
      call libgint_internal_dispatch( handle )
   end subroutine libgint_dispatch

end module libgint























