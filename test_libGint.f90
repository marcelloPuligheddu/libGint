program test_libGint
   
   USE iso_C_binding, ONLY: c_ptr
   USE libGint, ONLY: libgint_init

   TYPE(c_ptr), SAVE :: libGint_handle

   !$omp parallel
   CALL libgint_init(libGint_handle)
   !$omp end parallel

end program test_libGint
