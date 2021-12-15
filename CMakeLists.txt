cmake_minimum_required(VERSION 3.16)
project(ConformalIdealDelaunay)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(DIR_EXT "${CMAKE_CURRENT_SOURCE_DIR}/ext")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/util")

# enable CGAL if MPFR is needed
option(LIBIGL_WITH_CGAL                "Use CGAL"           ON)
option(RENDER_TEXTURE                  "Render results"     ON)
# enable embree for speeding up texture rendering
option(LIBIGL_WITH_EMBREE              "Use EMBREE"         ON)

if (RENDER_TEXTURE)

  option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
  option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

  add_library(plot src/util/plot.cpp)
  target_link_libraries(plot PUBLIC igl::core igl::opengl_glfw)

endif() 

set(LIBIGL_INCLUDE_DIR ${DIR_EXT}/libigl/include)
find_package(LIBIGL REQUIRED QUIET)

add_compile_definitions(PYBIND)
add_subdirectory(${DIR_EXT}/pybind11)
add_subdirectory(${DIR_EXT}/spdlog)

# for comparison - wrapper for comformal-similarity-map
add_executable(CSM_bin src/conformal_seamless_similarity/CSM.cpp)
target_link_libraries(CSM_bin PUBLIC igl::core)

# added for M_PI definition
add_compile_definitions(_USE_MATH_DEFINES)

# load MPFR through cgal
if (LIBIGL_WITH_CGAL)
  add_compile_definitions(WITH_MPFR)
  include_directories(${MPFR_INCLUDE_DIR})
  include_directories(${GMP_INCLUDE_DIR})
  link_directories(${MPFR_LIBRARIES_DIR})
endif()

# only wrap cpp impl - could be used directly w/o pybind
add_library(conformal_cpp SHARED   
  src/conformal_ideal_delaunay/Halfedge.cc
  src/conformal_ideal_delaunay/OverlayMesh.cc
  src/conformal_ideal_delaunay/BarycenterMapping.cc
)

# build pybind functions
add_library(conformal_py MODULE
  src/conformal_ideal_delaunay/Halfedge.cc
  src/conformal_ideal_delaunay/OverlayMesh.cc
  src/conformal_ideal_delaunay/BarycenterMapping.cc
  src/conformal_ideal_delaunay/ConformalPybind.cpp
)

if (RENDER_TEXTURE)
  add_compile_definitions(RENDER_TEXTURE)
  if (LIBIGL_WITH_EMBREE)
    add_compile_definitions(USE_EMBREE)
    target_link_libraries(conformal_py PRIVATE igl::embree plot)
  else()
    target_link_libraries(conformal_py PRIVATE plot)
  endif()
endif()

target_link_libraries(conformal_cpp PUBLIC igl::core spdlog::spdlog ${MPFR_LIBRARIES})
target_link_libraries(conformal_py PUBLIC igl::core pybind11::module spdlog::spdlog ${MPFR_LIBRARIES})
set_target_properties(conformal_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/py)
set_target_properties(conformal_py PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")

if (MSVC)
  set_target_properties(conformal_py PROPERTIES SUFFIX ".pyd")
  
  # need to copy dependent dlls to the same folder as conformal_py
  get_target_property(other_libs conformal_py INTERFACE_LINK_LIBRARIES)
  set(locations)
  list(APPEND locations ${main_lib} ${other_libs})
  foreach(location ${locations})
    string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${location})
    set(location "${CMAKE_MATCH_1}.dll")
    if(EXISTS "${location}" AND location MATCHES "^.*\\.dll$")
      add_custom_command(TARGET conformal_py POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${location}" $<TARGET_FILE_DIR:conformal_py>)
    endif()
  endforeach()

endif()
