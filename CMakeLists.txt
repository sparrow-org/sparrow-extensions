# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.28)

# This is better specified per target, but cmake keeps ignoring these language version
# specification when building this project by itself, in particular the gnu extensions,
# so here we go.
# This will affects all following targets being defined.
set(CMAKE_CXX_EXTENSIONS OFF)
cmake_policy(SET CMP0091 NEW)

# MSVC debug information format flags are selected via
# CMAKE_MSVC_DEBUG_INFORMATION_FORMAT, instead of
# embedding flags in e.g. CMAKE_CXX_FLAGS_RELEASE.
# New in CMake 3.25.
#
# Supports debug info with SCCache
# (https://github.com/mozilla/sccache?tab=readme-ov-file#usage)
# avoiding “fatal error C1041: cannot open program database; if
# multiple CL.EXE write to the same .PDB file, please use /FS"
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
cmake_policy(SET CMP0141 NEW)

project(sparrow-extensions CXX)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()
message(STATUS "🔧 C++ standard: ${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) # We don't use modules
include(CMakeDependentOption)

set(SPARROW_EXTENSIONS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SPARROW_EXTENSIONS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(CMAKE_CXX_COMPILER_LAUNCHER)
    message(STATUS "Using C++ compiler launcher: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
message(DEBUG "CMake module path: ${CMAKE_MODULE_PATH}")

# Versionning
# ===========
file(STRINGS "${SPARROW_EXTENSIONS_INCLUDE_DIR}/sparrow_extensions/config/sparrow_extensions_version.hpp" sparrow_extensions_version_defines
    REGEX "constexpr int SPARROW_EXTENSIONS_VERSION_(MAJOR|MINOR|PATCH)")

foreach(ver ${sparrow_extensions_version_defines})
    if(ver MATCHES "constexpr int SPARROW_EXTENSIONS_VERSION_(MAJOR|MINOR|PATCH) = ([0-9]+);$")
        set(PROJECT_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()

set(CMAKE_PROJECT_VERSION
    ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

message(STATUS "Building sparrow-extensions v${CMAKE_PROJECT_VERSION}")

# Binary version
# See the following URL for explanations about the binary versionning
# https://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html#Updating-version-info
file(STRINGS "${SPARROW_EXTENSIONS_INCLUDE_DIR}/sparrow_extensions/config/sparrow_extensions_version.hpp" sparrow_extensions_version_defines
    REGEX "constexpr int SPARROW_EXTENSIONS_BINARY_(CURRENT|REVISION|AGE)")

foreach(ver ${sparrow_extensions_version_defines})
    if(ver MATCHES "constexpr int SPARROW_EXTENSIONS_BINARY_(CURRENT|REVISION|AGE) = ([0-9]+);$")
        set(SPARROW_EXTENSIONS_BINARY_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()

set(SPARROW_EXTENSIONS_BINARY_VERSION
    ${SPARROW_EXTENSIONS_BINARY_CURRENT}.${SPARROW_EXTENSIONS_BINARY_REVISION}.${SPARROW_EXTENSIONS_BINARY_AGE})

message(STATUS "sparrow-extensions binary version: v${SPARROW_EXTENSIONS_BINARY_VERSION}")

# Build options
# =============
option(SPARROW_EXTENSIONS_BUILD_TESTS "Build sparrow-extensions test suite" OFF)
message(STATUS "🔧 Build tests: ${SPARROW_EXTENSIONS_BUILD_TESTS}")
option(BUILD_BENCHMARKS "Build sparrow-extensions benchmark suite" OFF)
message(STATUS "🔧 Build benchmarks: ${BUILD_BENCHMARKS}")
option(SPARROW_EXTENSIONS_BUILD_DOCS "Build sparrow-extensions documentation" OFF)
message(STATUS "🔧 Build docs: ${SPARROW_EXTENSIONS_BUILD_DOCS}")
option(BUILD_EXAMPLES "Build sparrow-extensions examples" OFF)
message(STATUS "🔧 Build examples: ${BUILD_EXAMPLES}")
option(USE_DATE_POLYFILL "Use date polyfill implementation" ON)
message(STATUS "🔧 Use date polyfill: ${USE_DATE_POLYFILL}")
option(ENABLE_INTEGRATION_TEST "Enable integration tests" OFF)
message(STATUS "🔧 Enable integration tests: ${ENABLE_INTEGRATION_TEST}")
option(CREATE_JSON_READER_TARGET "Create json_reader target, automatically set when ENABLE_INTEGRATION_TEST is ON" OFF)

if(ENABLE_INTEGRATION_TEST)
    set(CREATE_JSON_READER_TARGET ON)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4 OR MSVC)
    set(DEFAULT_USE_LARGE_INT_PLACEHOLDERS ON)
else()
    set(DEFAULT_USE_LARGE_INT_PLACEHOLDERS OFF)
endif()

set(CMAKE_DEBUG_POSTFIX "d")

option(ENABLE_COVERAGE "Enable test coverage" OFF)
message(STATUS "🔧 Enable coverage: ${ENABLE_COVERAGE}")

if(ENABLE_COVERAGE)
    include(code_coverage)
endif()

include(CheckCXXSymbolExists)

if(cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(header version)
else()
    set(header ciso646)
endif()

check_cxx_symbol_exists(_LIBCPP_VERSION ${header} LIBCPP)

if(LIBCPP)
    message(STATUS "Using libc++")

    # Allow the use of not visible yet availabile features, such
    # as some formatter for new types.
    add_compile_definitions(_LIBCPP_DISABLE_AVAILABILITY)
endif()

# Linter options
# =============
option(ACTIVATE_LINTER "Create targets to run clang-format" OFF)
message(STATUS "🔧 Activate linter: ${ACTIVATE_LINTER}")
cmake_dependent_option(ACTIVATE_LINTER_DURING_COMPILATION "Run linter during the compilation" ON "ACTIVATE_LINTER" OFF)

if(ACTIVATE_LINTER)
    include(clang-format)
    include(clang-tidy)
endif()

# Dependencies
# ============

include(external_dependencies)

set(SPARROW_EXTENSIONS_INTERFACE_DEPENDENCIES "" CACHE STRING "List of dependencies to be linked to the sparrow-extensions target")
set(SPARROW_EXTENSIONS_COMPILE_DEFINITIONS "" CACHE STRING "List of public compile definitions of the sparrow-extensions target")

if(SPARROW_EXTENSIONS_CONTRACTS_THROW_ON_FAILURE)
    message(STATUS "Using contract exceptions")
    list(APPEND SPARROW_EXTENSIONS_COMPILE_DEFINITIONS SPARROW_EXTENSIONS_CONTRACTS_THROW_ON_FAILURE=1)
endif()

# Build
# =====
set(BINARY_BUILD_DIR "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BINARY_BUILD_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BINARY_BUILD_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${BINARY_BUILD_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${BINARY_BUILD_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${BINARY_BUILD_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${BINARY_BUILD_DIR}")

set(SPARROW_EXTENSIONS_HEADERS
    # config
    ${SPARROW_EXTENSIONS_INCLUDE_DIR}/sparrow_extensions/config/config.hpp
    ${SPARROW_EXTENSIONS_INCLUDE_DIR}/sparrow_extensions/config/sparrow_extensions_version.hpp

    # ./
    ${SPARROW_EXTENSIONS_INCLUDE_DIR}/sparrow_extensions/uuid_array.hpp

    #../
    # ${SPARROW_EXTENSIONS_INCLUDE_DIR}/sparrow_extensions.hpp
)

set(SPARROW_EXTENSIONS_SRC
    # ${SPARROW_EXTENSIONS_SOURCE_DIR}/uuid_array.cpp
)

option(SPARROW_EXTENSIONS_BUILD_SHARED "Build sparrow-extensions as a shared library" ON)

if(SPARROW_EXTENSIONS_BUILD_SHARED)
    message(STATUS "🔧 Build shared library")
    set(SPARROW_EXTENSIONS_LIBRARY_TYPE SHARED)
else()
    message(STATUS "🔧 Build static library")
    set(SPARROW_EXTENSIONS_LIBRARY_TYPE STATIC)
    list(APPEND SPARROW_EXTENSIONS_COMPILE_DEFINITIONS SPARROW_EXTENSIONS_STATIC_LIB)
endif()

add_library(sparrow-extensions ${SPARROW_EXTENSIONS_LIBRARY_TYPE} ${SPARROW_EXTENSIONS_SRC} ${SPARROW_EXTENSIONS_HEADERS})

# Set linker language explicitly since there may be no source files yet
set_target_properties(sparrow-extensions PROPERTIES LINKER_LANGUAGE CXX)

target_compile_definitions(sparrow-extensions PUBLIC ${SPARROW_EXTENSIONS_COMPILE_DEFINITIONS})

set_property(TARGET sparrow-extensions PROPERTY POSITION_INDEPENDENT_CODE ON)

if(UNIX)
    # CMake does not compute the version number of so files as libtool
    # does on Linux. Strictly speaking, we should exclude FreeBSD and
    # Apple from this, but that would require having different version
    # numbers depending on the platform. We prefer to follow the
    # libtool pattern everywhere.
    math(EXPR SPARROW_EXTENSIONS_BINARY_COMPATIBLE "${SPARROW_EXTENSIONS_BINARY_CURRENT} - ${SPARROW_EXTENSIONS_BINARY_AGE}")
    set_target_properties(
        sparrow-extensions
        PROPERTIES
        VERSION "${SPARROW_EXTENSIONS_BINARY_COMPATIBLE}.${SPARROW_EXTENSIONS_BINARY_REVISION}.${SPARROW_EXTENSIONS_BINARY_AGE}"
        SOVERSION ${SPARROW_EXTENSIONS_BINARY_COMPATIBLE}
    )
    target_compile_options(sparrow-extensions PRIVATE "-fvisibility=hidden")
else()
    set_target_properties(
        sparrow-extensions
        PROPERTIES
        VERSION ${SPARROW_EXTENSIONS_BINARY_VERSION}
        SOVERSION ${SPARROW_EXTENSIONS_BINARY_CURRENT}
    )
    target_compile_definitions(sparrow-extensions PRIVATE SPARROW_EXTENSIONS_EXPORTS)
endif()

target_include_directories(sparrow-extensions PUBLIC
    $<BUILD_INTERFACE:${SPARROW_EXTENSIONS_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

set_target_properties(sparrow-extensions PROPERTIES CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(sparrow-extensions PUBLIC cxx_std_20)
target_link_libraries(sparrow-extensions PUBLIC sparrow::sparrow ${SPARROW_EXTENSIONS_INTERFACE_DEPENDENCIES})

if(ENABLE_COVERAGE)
    enable_coverage(sparrow-extensions)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    target_compile_options(sparrow-extensions
        PUBLIC
        -Wa,-mbig-obj)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(EMSCRIPTEN)
        target_compile_options(sparrow-extensions
            PRIVATE
            -fPIC)
    else()
        target_compile_options(sparrow-extensions
            PUBLIC
            -Wno-c99-extensions)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(sparrow-extensions
        PUBLIC
        /bigobj
    )
endif()

if(ENABLE_INTEGRATION_TEST)
    add_subdirectory(c_data_integration_tests)
endif()

if(SPARROW_EXTENSIONS_BUILD_TESTS)
    message(STATUS "🧪 Create tests targets")
    enable_testing()
    add_subdirectory(tests)
endif()

# Docs
# ====
if(SPARROW_EXTENSIONS_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Examples
# ========
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
# ============
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install target
set(SPARROW_EXTENSIONS_EXPORTED_TARGETS sparrow-extensions)

install(TARGETS ${SPARROW_EXTENSIONS_EXPORTED_TARGETS}
    EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
    NAMESPACE sparrow-extensions::)

# Install headers
install(DIRECTORY ${SPARROW_EXTENSIONS_INCLUDE_DIR}/sparrow_extensions
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN ".clang-*" EXCLUDE
    PATTERN "README.md" EXCLUDE)

install(FILES ${SPARROW_EXTENSIONS_INCLUDE_DIR}/sparrow_extensions.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install CMake configuration files
set(SPARROW_EXTENSIONS_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for sparrow-extensionsConfig.cmake")

configure_package_config_file(cmake/${PROJECT_NAME}Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${SPARROW_EXTENSIONS_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${SPARROW_EXTENSIONS_CMAKECONFIG_INSTALL_DIR})

install(EXPORT ${PROJECT_NAME}-targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE sparrow-extensions::
    DESTINATION ${SPARROW_EXTENSIONS_CMAKECONFIG_INSTALL_DIR})
