# Copyright 2024 Man Group Operations Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.28)

set(CMAKE_ENABLE_EXPORTS TRUE)
set(MAKE_EXECUTABLE_ENABLE_EXPORTS TRUE)

enable_testing()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(sparrow_extensions-test CXX)
    find_package(sparrow REQUIRED CONFIG)
    set(SPARROW_INCLUDE_DIR ${sparrow_INCLUDE_DIRS})
endif()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting tests build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
else()
    message(STATUS "Tests build type is ${CMAKE_BUILD_TYPE}")
endif()
  
set(SPARROW_EXTENSIONS_TESTS_SOURCES
    main.cpp
    test_bool8_array.cpp
    test_json_array.cpp
    test_uuid_array.cpp
    metadata_sample.hpp
)

set(test_target "test_sparrow_extensions_lib")
add_executable(${test_target} ${SPARROW_EXTENSIONS_TESTS_SOURCES})

if(EMSCRIPTEN)
    target_compile_options(${test_target}
        PUBLIC "SHELL: -fexceptions"
        PUBLIC "SHELL: -fPIC"
    )

    target_link_libraries(${test_target}
        PRIVATE
            sparrow
            doctest::doctest
            better_junit_reporter)

    target_link_options(${test_target}
        PUBLIC "SHELL: -fexceptions"
        PUBLIC "SHELL: -fPIC"
        PUBLIC "SHELL: -s MAIN_MODULE=1"
        PUBLIC "SHELL: -s WASM_BIGINT"
        PUBLIC "SHELL: -s ASSERTIONS=0"
        PUBLIC "SHELL: -s ALLOW_MEMORY_GROWTH=1"
        PUBLIC "SHELL: -s STACK_SIZE=32mb"
        PUBLIC "SHELL: -s INITIAL_MEMORY=128mb"
        PUBLIC "SHELL: --preload-file ${SYSROOT_PATH}/include@/include"
    )
else()
    set(TEST_LINK_LIBRARIES
        sparrow-extensions
        doctest::doctest
    )

    if(TARGET better_junit_reporter)
        list(APPEND TEST_LINK_LIBRARIES better_junit_reporter)
        set(HAS_BETTER_JUNIT_REPORTER TRUE)
    else()
        set(HAS_BETTER_JUNIT_REPORTER FALSE)
    endif()

    target_link_libraries(${test_target}
        PUBLIC
            ${TEST_LINK_LIBRARIES})

    target_compile_definitions(${test_target}
        PRIVATE
            $<$<BOOL:${HAS_BETTER_JUNIT_REPORTER}>:HAS_BETTER_JUNIT_REPORTER>)

    include(compile_options)

    target_compile_options(${test_target}
        PUBLIC
            ${compile_options}
            $<$<BOOL:USE_SANITIZER>:${SANITIZER_COMPILE_OPTIONS}>)

    target_link_options(${test_target}
        PRIVATE
            $<$<BOOL:USE_SANITIZER>:${SANITIZER_LINK_OPTIONS}>)

    target_compile_definitions(${test_target}
        PRIVATE
            DOCTEST_CONFIG_VOID_CAST_EXPRESSIONS)

    if(ENABLE_COVERAGE)
        enable_coverage(${test_target})
    endif()

    # We do not use non-standard C++
    set_target_properties(${test_target} 
        PROPERTIES
            CMAKE_CXX_EXTENSIONS OFF)
    target_compile_features(${test_target} PRIVATE cxx_std_20)

    add_custom_target(run_sparrow_extensions_tests
        COMMAND ${test_target}
        DEPENDS 
            ${test_target}
        COMMENT "Running tests for Doctest"
        USES_TERMINAL
    )

    set_target_properties(run_sparrow_extensions_tests PROPERTIES FOLDER "Tests utilities")
endif()
