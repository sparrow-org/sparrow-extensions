JSON Array               {#json_array}
==========

Introduction
------------

The JSON array is an Arrow-compatible array for storing JSON-encoded data according to the [Apache Arrow canonical extension specification for JSON](https://arrow.apache.org/docs/format/CanonicalExtensions.html#json).

Each element is stored as a UTF-8 encoded string containing valid JSON data.

The JSON extension type is defined as:
- Extension name: `arrow.json`
- Storage type: `String` (Utf8), `LargeString` (LargeUtf8), or `StringView` (Utf8View)
- Extension metadata: none

Three variants are provided to accommodate different use cases:

| Type | Storage Type | Description |
| ---- | ------------ | ----------- |
| `json_array` | String (32-bit offsets) | Standard choice for most JSON datasets |
| `big_json_array` | LargeString (64-bit offsets) | For datasets exceeding 2GB cumulative string length |
| `json_view_array` | StringView | View-based storage for optimized performance |

Usage
-----

### Basic Usage

```cpp
#include "sparrow_extensions/json_array.hpp"
using namespace sparrow;

// Create a JSON array from string values
std::vector<std::string> json_values = {
    R"({"name": "Alice", "age": 30})",
    R"({"name": "Bob", "age": 25})",
    R"([1, 2, 3, 4, 5])",
    R"("simple string")",
    R"(null)"
};
json_array arr(json_values);

// Access elements
for (size_t i = 0; i < arr.size(); ++i)
{
    if (arr[i].has_value())
    {
        std::string_view json = arr[i].value();
        // Parse or process JSON...
    }
}
```

### With Nullable Values

```cpp
#include "sparrow_extensions/json_array.hpp"
using namespace sparrow;

// Create a JSON array with null values (distinct from JSON "null")
std::vector<nullable<std::string>> values = {
    nullable<std::string>(R"({"key": "value"})"),
    nullable<std::string>(),  // null (missing value)
    nullable<std::string>(R"(null)"),  // JSON null (present value)
    nullable<std::string>(R"([])"),
};
json_array arr(values);
```

### Choosing the Right Variant

#### json_array (32-bit offsets)

Use for most JSON datasets where the cumulative length of all strings is less than 2GB:

```cpp
#include "sparrow_extensions/json_array.hpp"
using namespace sparrow;

std::vector<std::string> values = {/* ... */};
json_array arr(values);  // 32-bit offsets, up to 2GB total
```

#### big_json_array (64-bit offsets)

Use for very large JSON datasets where the cumulative length may exceed 2GB:

```cpp
#include "sparrow_extensions/json_array.hpp"
using namespace sparrow;

std::vector<std::string> large_values = {/* ... */};
big_json_array arr(large_values);  // 64-bit offsets, unlimited size
```

#### json_view_array (view-based storage)

Use for optimized performance with the Binary View layout, which stores short values inline:

```cpp
#include "sparrow_extensions/json_array.hpp"
using namespace sparrow;

std::vector<std::string> values = {/* ... */};
json_view_array arr(values);  // View-based, optimal for mixed sizes
```

### Extension Metadata

All JSON array variants automatically set the following Arrow extension metadata:

- `ARROW:extension:name`: `"arrow.json"`
- `ARROW:extension:metadata`: `""`

This metadata is added to the Arrow schema, allowing other Arrow implementations to recognize the array as containing JSON data.

API Reference
-------------

### json_extension

The `json_extension` type alias is defined as `simple_extension<"arrow.json">`, providing the extension type implementation for all JSON array variants.

### json_array

```cpp
using json_array = variable_size_binary_array_impl<
    arrow_traits<std::string>::value_type,
    arrow_traits<std::string>::const_reference,
    std::int32_t,
    json_extension>;
```

| Feature | Description |
| ------- | ----------- |
| Storage type | String (Utf8) |
| Offset type | 32-bit (`int32_t`) |
| Max cumulative size | 2^31-1 bytes (~2GB) |
| Extension name | `"arrow.json"` |

### big_json_array

```cpp
using big_json_array = variable_size_binary_array_impl<
    arrow_traits<std::string>::value_type,
    arrow_traits<std::string>::const_reference,
    std::int64_t,
    json_extension>;
```

| Feature | Description |
| ------- | ----------- |
| Storage type | LargeString (LargeUtf8) |
| Offset type | 64-bit (`int64_t`) |
| Max cumulative size | 2^63-1 bytes |
| Extension name | `"arrow.json"` |

### json_view_array

```cpp
using json_view_array = variable_size_binary_view_array_impl<
    arrow_traits<std::string>::value_type,
    arrow_traits<std::string>::const_reference,
    json_extension>;
```

| Feature | Description |
| ------- | ----------- |
| Storage type | StringView (Utf8View) |
| Layout | Binary View (inline short strings) |
| Extension name | `"arrow.json"` |
