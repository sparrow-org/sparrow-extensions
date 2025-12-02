Installation         {#installation}
============

From source with conda/mamba
----------------------------

Sparrow Extensions has a few dependencies that you can install in a mamba environment:

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

From source with cmake FetchContent
-----------------------------------

You can also use CMake's FetchContent to include sparrow-extensions in your project:

```cmake
include(FetchContent)

FetchContent_Declare(
    sparrow-extensions
    GIT_REPOSITORY https://github.com/Alex-PLACET/sparrow-extensions.git
    GIT_TAG main
)

FetchContent_MakeAvailable(sparrow-extensions)

target_link_libraries(your_target PRIVATE sparrow-extensions::sparrow-extensions)
```

Dependencies
------------

Sparrow Extensions depends on:

- [sparrow](https://github.com/man-group/sparrow) - C++20 implementation of the Apache Arrow Columnar Format

