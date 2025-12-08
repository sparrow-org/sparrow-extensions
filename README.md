# sparrow-extensions

[![GHA Linux](https://github.com/QuantStack/sparrow-extensions/actions/workflows/linux.yml/badge.svg)](https://github.com/QuantStack/sparrow-extensions/actions/workflows/linux.yml)
[![GHA OSX](https://github.com/QuantStack/sparrow-extensions/actions/workflows/osx.yml/badge.svg)](https://github.com/QuantStack/sparrow-extensions/actions/workflows/osx.yml)
[![GHA Windows](https://github.com/QuantStack/sparrow-extensions/actions/workflows/windows.yml/badge.svg)](https://github.com/QuantStack/sparrow-extensions/actions/workflows/windows.yml)
[![GHA Docs](https://github.com/QuantStack/sparrow-extensions/actions/workflows/deploy-pages.yaml/badge.svg)](https://github.com/QuantStack/sparrow-extensions/actions/workflows/deploy-pages.yaml)

Extension types for the [sparrow](https://github.com/man-group/sparrow) library

## Introduction

`sparrow-extensions` provides additional array types and extension implementations for the [sparrow](https://github.com/man-group/sparrow) library, following the [Apache Arrow Canonical Extensions](https://arrow.apache.org/docs/format/CanonicalExtensions.html) specification.

`sparrow-extensions` requires a modern C++ compiler supporting C++20.

### Available Extensions

- `uuid_array`: Arrow-compatible array for storing UUID values as 16-byte fixed-width binary according to the `arrow.uuid` extension type specification.
- `json_array`: Arrow-compatible array for storing JSON values as UTF-8 strings according to the `arrow.json` extension type specification.
- `bool8_array`: Arrow-compatible array for storing boolean values as 8-bit integers according to the `arrow.bool8` extension type specification.
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
    -DSPARROW_EXTENSIONS_BUILD_TESTS=ON
make install
```

## Usage

### Requirements

Compilers:
- Clang 18 or higher
- GCC 11.2 or higher
- Apple Clang 16 or higher
- MSVC 19.41 or higher

## Dependencies

- [sparrow](https://github.com/man-group/sparrow) - C++20 implementation of the Apache Arrow Columnar Format

## License

This software is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
