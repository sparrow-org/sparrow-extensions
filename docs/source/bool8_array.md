Bool8 Array               {#bool8_array}
===========

Introduction
------------

The Bool8 array is an Arrow-compatible array for storing boolean values using 8-bit storage according to the [Apache Arrow canonical extension specification for Bool8](https://arrow.apache.org/docs/format/CanonicalExtensions.html#bool8).

Bool8 represents a boolean value using 1 byte (8 bits) to store each value instead of only 1 bit as in the original Arrow Boolean type. Although less compact than the original representation, Bool8 may have better zero-copy compatibility with various systems that also store booleans using 1 byte.

The Bool8 extension type is defined as:
- Extension name: `arrow.bool8`
- Storage type: `Int8`
- `false` is denoted by the value `0`
- `true` can be specified using any non-zero value (preferably `1`)
- Extension metadata: empty string

Usage
-----

### Basic Usage

```cpp
#include "sparrow_extensions/bool8_array.hpp"
using namespace sparrow;

// Create a Bool8 array from a vector of bools
std::vector<bool> values = {true, false, true, true, false};
bool8_array arr(values);

// Access elements
for (size_t i = 0; i < arr.size(); ++i)
{
    if (arr[i].has_value())
    {
        bool val = arr[i].value();
        // Use val...
    }
}
```

### With Nullable Values

```cpp
#include "sparrow_extensions/bool8_array.hpp"
using namespace sparrow;

// Create a Bool8 array with null values
std::vector<nullable<bool>> values = {
    nullable<bool>(true),
    nullable<bool>(),        // null
    nullable<bool>(false),
    nullable<bool>(),        // null
    nullable<bool>(true)
};
bool8_array arr(values);

// Check for null values
for (size_t i = 0; i < arr.size(); ++i)
{
    if (arr[i].has_value())
    {
        std::cout << (arr[i].value() ? "true" : "false") << "\n";
    }
    else
    {
        std::cout << "null\n";
    }
}
```

### Formatting (C++20)

When `<format>` is available, Bool8 arrays can be formatted directly:

```cpp
#include "sparrow_extensions/bool8_array.hpp"
#include <format>
#include <iostream>

using namespace sparrow;

std::vector<bool> values = {true, false, true};
bool8_array arr(values);

// Output: "Bool8 array [3]: [true, false, true]"
std::cout << std::format("{}", arr) << "\n";
```

### Extension Metadata

The Bool8 array automatically sets the following Arrow extension metadata:

- `ARROW:extension:name`: `"arrow.bool8"`
- `ARROW:extension:metadata`: `""`

This metadata is added to the Arrow schema, allowing other Arrow implementations to recognize the array as containing Bool8 values.

API Reference
-------------

### bool8_array

The `bool8_array` type is an alias for `sparrow::primitive_array<int8_t, simple_extension<"arrow.bool8">, bool>`, providing all the functionality of a primitive array with Bool8-specific extension metadata and boolean value semantics.

| Feature | Description |
| ------- | ----------- |
| Storage type | `int8_t` |
| Value type | `bool` |
| Extension name | `"arrow.bool8"` |
| Null support | Yes, via validity bitmap |

### Formatting Support

When compiled with C++20 `<format>` support, the following are available:

| Function | Description |
| -------- | ----------- |
| `std::formatter<bool8_array>` | Formatter specialization for `std::format` |
| `operator<<(std::ostream&, const bool8_array&)` | Stream output operator |
