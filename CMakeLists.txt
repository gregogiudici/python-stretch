cmake_minimum_required(VERSION 3.15...3.26)

project(python_stretch LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING "
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# We need to include stretch folder for signalsmith-stretch's headers
include_directories(include ${CMAKE_CURRENT_SOURCE_DIR}/include)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  Signalsmith

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  src/signalsmith-bindings.cpp
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version
    OUTPUT_VARIABLE CLANG_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" CLANG_VERSION_MATCH "${CLANG_VERSION_OUTPUT}")
  set(CLANG_MAJOR_VERSION ${CMAKE_MATCH_1})
  set(CLANG_MINOR_VERSION ${CMAKE_MATCH_2})
  set(CLANG_PATCH_VERSION ${CMAKE_MATCH_3})
endif()

# Optimize Signalsmith library
if (MSVC)
    target_compile_options(Signalsmith PRIVATE /O2 /fp:fast)
    target_link_options(Signalsmith PRIVATE /GL /LTCG)
else()
  if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_MAJOR_VERSION EQUAL 16 AND CLANG_MINOR_VERSION EQUAL 0 AND CLANG_PATCH_VERSION EQUAL 0)
    target_compile_options(Signalsmith PRIVATE -O3)    
  else()
    target_compile_options(Signalsmith PRIVATE -O3 -ffast-math)
  endif()
  target_link_options(Signalsmith PRIVATE -flto)
endif()

# Skip if we are building for apple AND python is greater than 3.8
if((NOT APPLE) OR (Python_VERSION VERSION_GREATER_EQUAL "3.9"))
  # Create a stub file for type hints
  nanobind_add_stub(
    Signalsmith_stub
    MODULE Signalsmith
    OUTPUT Signalsmith.pyi
    MARKER_FILE py.typed
    PYTHON_PATH $<TARGET_FILE_DIR:Signalsmith>
    DEPENDS Signalsmith
  )
endif()


# Install directive for scikit-build-core
install(TARGETS Signalsmith LIBRARY DESTINATION python_stretch)

if((NOT APPLE) OR (Python_VERSION VERSION_GREATER_EQUAL "3.9"))
  # Install stub files
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed ${CMAKE_CURRENT_BINARY_DIR}/Signalsmith.pyi DESTINATION python_stretch)
endif()

