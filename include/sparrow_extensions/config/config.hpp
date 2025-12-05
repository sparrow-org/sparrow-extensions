// Copyright 2024 Man Group Operations Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// DLL export/import macros for Windows
#if defined(_WIN32)
#    if defined(SPARROW_EXTENSIONS_STATIC_LIB)
#        define SPARROW_EXTENSIONS_API
#    elif defined(SPARROW_EXTENSIONS_EXPORTS)
#        define SPARROW_EXTENSIONS_API __declspec(dllexport)
#    else
#        define SPARROW_EXTENSIONS_API __declspec(dllimport)
#    endif
#else
#    define SPARROW_EXTENSIONS_API __attribute__((visibility("default")))
#endif

// If using gcc version < 12, we define the constexpr keyword to be empty.
#if defined(__GNUC__) && __GNUC__ < 12
#    define SPARROW_EXTENSIONS_CONSTEXPR_GCC_11 inline
#else
#    define SPARROW_EXTENSIONS_CONSTEXPR_GCC_11 constexpr
#endif

#if (!defined(__clang__) && defined(__GNUC__))
#    if (__GNUC__ < 12 && __GNUC_MINOR__ < 3)
#        define SPARROW_EXTENSIONS_GCC_11_2_WORKAROUND 1
#    endif
#endif

// If using clang or apple-clang version < 18 or clang 18 on Android, we define the constexpr keyword to be
// "inline".
#if defined(__clang__) && ((__clang_major__ < 18) || (__clang_major__ == 18 && defined(__ANDROID__)))
#    define SPARROW_EXTENSIONS_CONSTEXPR_CLANG inline
#else
#    define SPARROW_EXTENSIONS_CONSTEXPR_CLANG constexpr
#endif

#if defined(__EMSCRIPTEN__) && defined(__CLANG_REPL__)
#    include <clang/Interpreter/CppInterOp.h>

#    ifndef SPARROW_EXTENSIONS_USE_DATE_POLYFILL
#        define SPARROW_EXTENSIONS_USE_DATE_POLYFILL 1
#    endif

#    ifndef HALF_ERRHANDLING_THROWS
#        define HALF_ERRHANDLING_THROWS 1
#    endif

static bool _sparrow_loaded = []()
{
    Cpp::LoadLibrary("/lib/libsparrow-extensions.so", false);
    return true;
}();
#endif
