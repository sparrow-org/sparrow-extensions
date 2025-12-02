UUID Array               {#uuid_array}
==========

Introduction
------------

The UUID array is an Arrow-compatible array for storing UUID values according to the [Apache Arrow canonical extension specification for UUIDs](https://arrow.apache.org/docs/format/CanonicalExtensions.html#uuid).

Each UUID is stored as a 16-byte (128-bit) fixed-width binary value.

The UUID extension type is defined as:
- Extension name: `arrow.uuid`
- Storage type: `FixedSizeBinary(16)`
- Extension metadata: none

Usage
-----

### Basic Usage

```cpp
#include "sparrow_extensions/uuid_array.hpp"
namespace spx = sparrow_extensions;

// Create a UUID array
spx::uuid_array uuids = /* ... */;

// UUIDs are stored as 16-byte fixed-width binary values
// with extension metadata "ARROW:extension:name" = "arrow.uuid"
```

### Extension Metadata

The UUID array automatically sets the following Arrow extension metadata:

- `ARROW:extension:name`: `"arrow.uuid"`
- `ARROW:extension:metadata`: `""`

This metadata is added to the Arrow schema, allowing other Arrow implementations to recognize the array as containing UUID values.

API Reference
-------------

### uuid_extension

The `uuid_extension` struct provides the extension type implementation:

| Member | Description |
| ------ | ----------- |
| `UUID_SIZE` | Constant equal to 16 (bytes per UUID) |
| `EXTENSION_NAME` | Constant equal to `"arrow.uuid"` |

### uuid_array

The `uuid_array` type is an alias for `sparrow::fixed_width_binary_array_impl` with the `uuid_extension` mixin, providing all the functionality of a fixed-width binary array with UUID-specific extension metadata.

