# Try to find the MPFR libraries
# MPFR_FOUND - system has MPFR lib
# MPFR_INCLUDE_DIR - the MPFR include directory
# MPFR_LIBRARIES_DIR - Directory where the MPFR libraries are located
# MPFR_LIBRARIES - the MPFR libraries

INCLUDE(FindPackageHandleStandardArgs)

# GMP is also needed, prerequisite
find_path(GMP_INCLUDE_DIR
          NAMES gmp.h
          HINTS ENV GMP_INC_DIR
                ENV GMP_DIR
                ENV GMP_INC
          PATH_SUFFIXES include
          DOC "The directory containing the GMP header files"
)

find_library(GMP_LIBRARIES NAMES gmp libgmp
  HINTS ENV GMP_LIB_DIR
        ENV GMP_DIR
        ENV GMP_LIB
  PATH_SUFFIXES lib
  DOC "Path to the GMP library"
)


# find MPFR
find_path(MPFR_INCLUDE_DIR
          NAMES mpfr.h
          HINTS ENV MPFR_INC_DIR
                ENV MPFR_DIR
                ENV MPFR_INC
          PATH_SUFFIXES include
          DOC "The directory containing the MPFR header files"
)

find_library(MPFR_LIBRARIES NAMES mpfr libmpfr-4 libmpfr-1
  HINTS ENV MPFR_LIB_DIR
        ENV MPFR_DIR
        ENV MPFR_LIB
  PATH_SUFFIXES lib
  DOC "Path to the MPFR library"
)

if ( MPFR_LIBRARIES AND GMP_LIBRARIES )
  get_filename_component(MPFR_LIBRARIES_DIR ${MPFR_LIBRARIES} PATH CACHE )
  get_filename_component(GMP_LIBRARIES_DIR ${GMP_LIBRARIES} PATH CACHE )
endif()

find_package_handle_standard_args(MPFR "DEFAULT_MSG" MPFR_LIBRARIES MPFR_INCLUDE_DIR)
find_package_handle_standard_args(GMP "DEFAULT_MSG" GMP_LIBRARIES GMP_INCLUDE_DIR)
