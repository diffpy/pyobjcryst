cmake_minimum_required(VERSION 3.15)

project(pyobjcryst CXX)

# Require C++17 globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Python / nanobind
# ---------------------------------------------------------------------------
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Boost (header-only: format, rational, date_time used by ObjCryst sources)
# Priority: explicit BOOST_INCLUDE_DIR, then the PyPI `libboost-headers`
# package (used for isolated pip builds), then system/conda headers.
# ---------------------------------------------------------------------------
if(BOOST_INCLUDE_DIR)
    set(PYOBJCRYST_BOOST_INCLUDE_DIR "${BOOST_INCLUDE_DIR}")
else()
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import importlib.util, os; s = importlib.util.find_spec('libboost_headers'); print(os.path.join(s.submodule_search_locations[0], 'include') if s else '')"
        OUTPUT_VARIABLE _PYPI_BOOST_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_PYPI_BOOST_INCLUDE_DIR AND EXISTS "${_PYPI_BOOST_INCLUDE_DIR}/boost/rational.hpp")
        set(PYOBJCRYST_BOOST_INCLUDE_DIR "${_PYPI_BOOST_INCLUDE_DIR}")
    else()
        find_path(BOOST_SYSTEM_INCLUDE_DIR boost/rational.hpp
            HINTS "$ENV{BOOST_ROOT}/include" "$ENV{Boost_ROOT}/include" "$ENV{CONDA_PREFIX}/include"
        )
        if(NOT BOOST_SYSTEM_INCLUDE_DIR)
            message(FATAL_ERROR
                "Boost headers not found. Install them with `pip install libboost-headers`, "
                "`conda install -c conda-forge libboost-headers`, or set -DBOOST_INCLUDE_DIR=<path>.")
        endif()
        set(PYOBJCRYST_BOOST_INCLUDE_DIR "${BOOST_SYSTEM_INCLUDE_DIR}")
    endif()
endif()
message(STATUS "Using Boost headers from: ${PYOBJCRYST_BOOST_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# External source paths
# ---------------------------------------------------------------------------
set(OBJCRYST_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/extern/objcryst/ObjCryst")
set(NEWMAT_SRC     "${CMAKE_CURRENT_SOURCE_DIR}/extern/newmat")
set(CCTBX_SRC      "${CMAKE_CURRENT_SOURCE_DIR}/extern/cctbx")

# ---------------------------------------------------------------------------
# Generate version.h so that ObjCryst sources can be compiled standalone
# ---------------------------------------------------------------------------
set(PYOBJCRYST_EMBED_VERSION_MAJOR 2024)
set(PYOBJCRYST_EMBED_VERSION_MINOR 1)
set(PYOBJCRYST_EMBED_VERSION_MICRO 0)
set(PYOBJCRYST_EMBED_VERSION_PATCH 3)
# Build integer: MAJOR*1e9 + MINOR*1e6 + MICRO*1e3 + PATCH
math(EXPR PYOBJCRYST_EMBED_VERSION
    "${PYOBJCRYST_EMBED_VERSION_MAJOR} * 1000000000 + \
     ${PYOBJCRYST_EMBED_VERSION_MINOR} * 1000000 + \
     ${PYOBJCRYST_EMBED_VERSION_MICRO} * 1000 + \
     ${PYOBJCRYST_EMBED_VERSION_PATCH}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/ObjCryst/version.h"
    @ONLY
)

# ---------------------------------------------------------------------------
# newmat static library
# ---------------------------------------------------------------------------
set(NEWMAT_SOURCES
    ${NEWMAT_SRC}/bandmat.cpp
    ${NEWMAT_SRC}/evalue.cpp
    ${NEWMAT_SRC}/myexcept.cpp
    ${NEWMAT_SRC}/newmat1.cpp
    ${NEWMAT_SRC}/newmat2.cpp
    ${NEWMAT_SRC}/newmat3.cpp
    ${NEWMAT_SRC}/newmat4.cpp
    ${NEWMAT_SRC}/newmat5.cpp
    ${NEWMAT_SRC}/newmat6.cpp
    ${NEWMAT_SRC}/newmat7.cpp
    ${NEWMAT_SRC}/newmat8.cpp
    ${NEWMAT_SRC}/newmat9.cpp
    ${NEWMAT_SRC}/newmatex.cpp
    ${NEWMAT_SRC}/sort.cpp
    ${NEWMAT_SRC}/submat.cpp
)

add_library(newmat STATIC ${NEWMAT_SOURCES})
target_include_directories(newmat PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/extern")
set_property(TARGET newmat PROPERTY POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# cctbx static library
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CCTBX_SOURCES
    "${CCTBX_SRC}/cctbx/eltbx/*.cpp"
    "${CCTBX_SRC}/cctbx/sgtbx/*.cpp"
    "${CCTBX_SRC}/cctbx/miller/*.cpp"
    "${CCTBX_SRC}/cctbx/uctbx/*.cpp"
)

add_library(cctbx STATIC ${CCTBX_SOURCES})
target_include_directories(cctbx PUBLIC
    "${CCTBX_SRC}/include"
    "${Python_ROOT_DIR}/include"
    "${PYOBJCRYST_BOOST_INCLUDE_DIR}"
)
set_property(TARGET cctbx PROPERTY POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# ObjCryst++ static library
# ---------------------------------------------------------------------------
file(GLOB OBJCRYST_SOURCES
    "${OBJCRYST_SRC}/ObjCryst/*.cpp"
    "${OBJCRYST_SRC}/RefinableObj/*.cpp"
    "${OBJCRYST_SRC}/CrystVector/*.cpp"
    "${OBJCRYST_SRC}/Quirks/VFNDebug.cpp"
    "${OBJCRYST_SRC}/Quirks/VFNStreamFormat.cpp"
    "${OBJCRYST_SRC}/Quirks/ci_string.cpp"
)
list(APPEND OBJCRYST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cpp")

add_library(ObjCryst STATIC ${OBJCRYST_SOURCES})

target_include_directories(ObjCryst PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/objcryst"
    "${CCTBX_SRC}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${PYOBJCRYST_BOOST_INCLUDE_DIR}"
)
set_property(TARGET ObjCryst PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(ObjCryst PUBLIC
    REAL=double
    BOOST_ERROR_CODE_HEADER_ONLY
    $<$<CONFIG:Debug>:__DEBUG__>
)
target_link_libraries(ObjCryst PUBLIC cctbx newmat)

# Put VFNDebug.cpp first so debug-level constants are defined before use
if(NOT MSVC)
    set_source_files_properties(
        "${OBJCRYST_SRC}/Quirks/VFNDebug.cpp"
        PROPERTIES COMPILE_FLAGS "-O0"
    )
endif()

# MSVC: same settings as the libobjcryst Windows build (/EHsc, M_PI etc.),
# plus /bigobj for the large binding translation units.
if(MSVC)
    foreach(_tgt newmat cctbx ObjCryst)
        target_compile_definitions(${_tgt} PUBLIC _USE_MATH_DEFINES NOMINMAX)
        target_compile_options(${_tgt} PUBLIC /EHsc /bigobj)
    endforeach()
endif()

# ---------------------------------------------------------------------------
# nanobind extension: _pyobjcryst
# ---------------------------------------------------------------------------
file(GLOB NB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/extensions/*.cpp")

nanobind_add_module(_pyobjcryst STABLE_ABI NB_STATIC ${NB_SOURCES})

target_include_directories(_pyobjcryst PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/objcryst"
    "${CCTBX_SRC}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${PYOBJCRYST_BOOST_INCLUDE_DIR}"
    ${Python_NumPy_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/extensions"
)

target_link_libraries(_pyobjcryst PRIVATE ObjCryst)

# Install into pyobjcryst package
install(TARGETS _pyobjcryst LIBRARY DESTINATION pyobjcryst)

# Licenses of the statically linked third-party code ship with the wheel.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/extern/objcryst/LICENSE.txt"
        DESTINATION pyobjcryst/licenses/objcryst)
install(FILES "${CCTBX_SRC}/LICENSE_2_0.txt" "${CCTBX_SRC}/COPYRIGHT_2_0.txt"
        DESTINATION pyobjcryst/licenses/cctbx)
install(FILES "${NEWMAT_SRC}/readme.txt" "${NEWMAT_SRC}/nm11.htm"
        DESTINATION pyobjcryst/licenses/newmat)
