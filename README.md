# sparrow-extensions

[![GHA Linux](https://github.com/Alex-PLACET/sparrow-extensions/actions/workflows/linux.yml/badge.svg)](https://github.com/Alex-PLACET/sparrow-extensions/actions/workflows/linux.yml)
[![GHA OSX](https://github.com/Alex-PLACET/sparrow-extensions/actions/workflows/osx.yml/badge.svg)](https://github.com/Alex-PLACET/sparrow-extensions/actions/workflows/osx.yml)
[![GHA Windows](https://github.com/Alex-PLACET/sparrow-extensions/actions/workflows/windows.yml/badge.svg)](https://github.com/Alex-PLACET/sparrow-extensions/actions/workflows/windows.yml)
[![GHA Docs](https://github.com/Alex-PLACET/sparrow-extensions/actions/workflows/deploy-pages.yaml/badge.svg)](https://github.com/Alex-PLACET/sparrow-extensions/actions/workflows/deploy-pages.yaml)

Extension types for the [sparrow](https://github.com/man-group/sparrow) library

## Introduction

`sparrow-extensions` provides additional array types and extension implementations for the [sparrow](https://github.com/man-group/sparrow) library, following the [Apache Arrow Canonical Extensions](https://arrow.apache.org/docs/format/CanonicalExtensions.html) specification.

`sparrow-extensions` requires a modern C++ compiler supporting C++20.

### Available Extensions

- **UUID Array**: Arrow-compatible array for storing UUID values as 16-byte fixed-width binary according to the `arrow.uuid` extension type specification.

## Installation

### Install from sources

`sparrow-extensions` has a few dependencies that you can install in a mamba environment:

```bash
mamba env create -f environment-dev.yml
mamba activate sparrow-extensions
```

You can then create a build directory, and build the project and install it with cmake:

```bash
mkdir build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DBUILD_TESTS=ON
make install
```

## Usage

### Requirements

Compilers:
- Clang 18 or higher
- GCC 11.2 or higher
- Apple Clang 16 or higher
- MSVC 19.41 or higher

### UUID Array

```cpp
#include "sparrow_extensions/uuid_array.hpp"
namespace spx = sparrow_extensions;

// Create a UUID array using the Arrow extension type
spx::uuid_array uuids = /* ... */;

// UUIDs are stored as 16-byte fixed-width binary values
// with extension metadata "ARROW:extension:name" = "arrow.uuid"
```

## Dependencies

- [sparrow](https://github.com/man-group/sparrow) - C++20 implementation of the Apache Arrow Columnar Format

## License

This software is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
