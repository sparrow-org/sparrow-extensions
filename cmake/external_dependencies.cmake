include(FetchContent)

option(FETCH_DEPENDENCIES_WITH_CMAKE "Fetch dependencies with CMake: Can be OFF, ON, or MISSING. If the latter, CMake will download only dependencies which are not previously found." OFF)
message(STATUS "🔧 FETCH_DEPENDENCIES_WITH_CMAKE: ${FETCH_DEPENDENCIES_WITH_CMAKE}")

if(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "OFF")
    set(FIND_PACKAGE_OPTIONS REQUIRED)
else()
    set(FIND_PACKAGE_OPTIONS QUIET)
endif()

function(find_package_or_fetch)
    set(options)
    set(oneValueArgs CONAN_PKG_NAME PACKAGE_NAME GIT_REPOSITORY TAG SOURCE_SUBDIR)
    set(multiValueArgs CMAKE_ARGS)
    cmake_parse_arguments(PARSE_ARGV 0 arg
        "${options}" "${oneValueArgs}" "${multiValueArgs}"
    )

    set(actual_pkg_name ${arg_PACKAGE_NAME})
    if(arg_CONAN_PKG_NAME)
        set(actual_pkg_name ${arg_CONAN_PKG_NAME})
    endif()

    if(NOT FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON")
        find_package(${actual_pkg_name} ${FIND_PACKAGE_OPTIONS})
    endif()

    if(arg_GIT_REPOSITORY)
        if(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON" OR FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "MISSING")
            if(NOT ${actual_pkg_name}_FOUND)
                message(STATUS "📦 Fetching ${arg_PACKAGE_NAME}")
                # Apply CMAKE_ARGS before fetching
                foreach(cmake_arg ${arg_CMAKE_ARGS})
                    string(REGEX MATCH "^([^=]+)=(.*)$" _ ${cmake_arg})
                    if(CMAKE_MATCH_1)
                        set(${CMAKE_MATCH_1} ${CMAKE_MATCH_2} CACHE BOOL "" FORCE)
                    endif()
                endforeach()
                set(fetch_args
                    ${arg_PACKAGE_NAME}
                    GIT_SHALLOW TRUE
                    GIT_REPOSITORY ${arg_GIT_REPOSITORY}
                    GIT_TAG ${arg_TAG}
                    GIT_PROGRESS TRUE
                    SYSTEM
                    EXCLUDE_FROM_ALL)
                if(arg_SOURCE_SUBDIR)
                    list(APPEND fetch_args SOURCE_SUBDIR ${arg_SOURCE_SUBDIR})
                endif()
                FetchContent_Declare(${fetch_args})
                FetchContent_MakeAvailable(${arg_PACKAGE_NAME})
                message(STATUS "\t✅ Fetched ${arg_PACKAGE_NAME}")
            else()
                message(STATUS "📦 ${actual_pkg_name} found here: ${${actual_pkg_name}_DIR}")
            endif()
        endif()
    else()
        # No GIT_REPOSITORY provided - only find_package is attempted
        if(${actual_pkg_name}_FOUND)
            message(STATUS "📦 ${actual_pkg_name} found here: ${${actual_pkg_name}_DIR}")
        elseif(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "OFF")
            message(FATAL_ERROR "Could not find ${actual_pkg_name} and no GIT_REPOSITORY provided for fetching")
        else()
            message(WARNING "Could not find ${actual_pkg_name} and no GIT_REPOSITORY provided for fetching")
        endif()
    endif()
endfunction()

set(SPARROW_BUILD_SHARED ${SPARROW_EXTENSIONS_BUILD_SHARED})
find_package_or_fetch(
    PACKAGE_NAME sparrow
    GIT_REPOSITORY https://github.com/man-group/sparrow.git
    TAG 2.0.0
)

if(NOT TARGET sparrow::sparrow)
    add_library(sparrow::sparrow ALIAS sparrow)
endif()

# add sparrow::sparrow to SPARROW_EXTENSIONS_INTERFACE_DEPENDENCIES list
set(SPARROW_EXTENSIONS_INTERFACE_DEPENDENCIES sparrow::sparrow)

find_package_or_fetch(
    PACKAGE_NAME simdjson
    GIT_REPOSITORY https://github.com/simdjson/simdjson.git
    TAG v4.2.4
)

set(SPARROW_EXTENSIONS_INTERFACE_DEPENDENCIES ${SPARROW_EXTENSIONS_INTERFACE_DEPENDENCIES} simdjson::simdjson)

if(SPARROW_EXTENSIONS_BUILD_TESTS)
    find_package_or_fetch(
        PACKAGE_NAME doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        TAG v2.4.12
    )

    # better_junit_reporter is provided by sparrow
    if(NOT TARGET better_junit_reporter)
        message(STATUS "📦 better_junit_reporter target provided by sparrow")
    endif()
endif()
