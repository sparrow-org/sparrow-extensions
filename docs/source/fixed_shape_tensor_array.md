Fixed Shape Tensor Array               {#fixed_shape_tensor_array}
===========================

Introduction
------------

The Fixed Shape Tensor Array is an Arrow-compatible array for storing multi-dimensional tensors with a fixed shape according to the [Apache Arrow canonical extension specification for FixedShapeTensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor).

This extension enables efficient storage and transfer of tensors (multi-dimensional arrays) with the same shape. Each element in the array represents a complete tensor of the specified shape. The underlying storage uses Arrow's `FixedSizeList` type to store flattened tensor data.

The FixedShapeTensor extension type is defined as:
- Extension name: `arrow.fixed_shape_tensor`
- Storage type: `FixedSizeList<T>` where `T` is the value type
- Extension metadata: JSON object containing shape, optional dimension names, and optional permutation
- List size: Product of all dimensions in the shape

Metadata Structure
------------------

The extension metadata is a JSON object with the following fields:

```json
{
  "shape": [dim0, dim1, ..., dimN],
  "dim_names": ["name0", "name1", ..., "nameN"],  // optional
  "permutation": [idx0, idx1, ..., idxN]          // optional
}
```

### Fields

- **shape** (required): Array of positive integers specifying the dimensions of each tensor
- **dim_names** (optional): Array of strings naming each dimension (must match shape length)
- **permutation** (optional): Array defining the physical-to-logical dimension mapping (must be a valid permutation of [0, 1, ..., N-1])

Usage
-----

### Basic Usage

```cpp
#include "sparrow_extensions/fixed_shape_tensor.hpp"
using namespace sparrow_extensions;
using namespace sparrow;

// Create 3 tensors of shape [2, 3] (2 rows, 3 columns)
std::vector<float> flat_data = {
    // First tensor
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,
    // Second tensor
    7.0f, 8.0f, 9.0f,
    10.0f, 11.0f, 12.0f,
    // Third tensor
    13.0f, 14.0f, 15.0f,
    16.0f, 17.0f, 18.0f
};

primitive_array<float> values_array(flat_data);
fixed_shape_tensor_extension::metadata tensor_meta{
    {2, 3},           // shape
    std::nullopt,     // no dimension names
    std::nullopt      // no permutation
};

const std::uint64_t list_size = tensor_meta.compute_size();  // 2 * 3 = 6

fixed_shape_tensor_array tensor_array(
    list_size,
    array(std::move(values_array)),
    tensor_meta
);

// Access properties
std::cout << "Number of tensors: " << tensor_array.size() << "\n";  // 3
std::cout << "Tensor shape: ";
for (auto dim : tensor_array.shape())
{
    std::cout << dim << " ";  // 2 3
}
std::cout << "\n";

// Access individual tensors
auto first_tensor = tensor_array[0];
if (first_tensor.has_value())
{
    // Process first tensor...
}
```

### With Dimension Names

```cpp
// Create a 3D tensor with named dimensions
std::vector<float> flat_data(100 * 200 * 500);  // Fill with data...

primitive_array<float> values_array(flat_data);
fixed_shape_tensor_extension::metadata tensor_meta{
    {100, 200, 500},                     // shape: channels, height, width
    std::vector<std::string>{"C", "H", "W"},  // dimension names
    std::nullopt                         // no permutation
};

const std::uint64_t list_size = tensor_meta.compute_size();

fixed_shape_tensor_array tensor_array(
    list_size,
    array(std::move(values_array)),
    tensor_meta
);

// Access metadata
const auto& meta = tensor_array.get_metadata();
if (meta.dim_names.has_value())
{
    for (const auto& name : *meta.dim_names)
    {
        std::cout << name << " ";  // C H W
    }
}
```

### With Permutation

```cpp
// Physical storage is [100, 200, 500] but logical layout is [500, 100, 200]
std::vector<float> flat_data(100 * 200 * 500);  // Fill with data...

primitive_array<float> values_array(flat_data);
fixed_shape_tensor_extension::metadata tensor_meta{
    {100, 200, 500},                     // physical shape
    std::nullopt,
    std::vector<std::int64_t>{2, 0, 1}   // permutation: logical[i] = physical[perm[i]]
};

const std::uint64_t list_size = tensor_meta.compute_size();

fixed_shape_tensor_array tensor_array(
    list_size,
    array(std::move(values_array)),
    tensor_meta
);

// Physical shape: [100, 200, 500]
// Logical shape: [500, 100, 200]
const auto& meta = tensor_array.get_metadata();
if (meta.permutation.has_value())
{
    std::cout << "Has permutation: ";
    for (auto idx : *meta.permutation)
    {
        std::cout << idx << " ";  // 2 0 1
    }
}
```

### With Array Name and Metadata

```cpp
// Create tensors with custom name and Arrow metadata
std::vector<double> flat_data(24);  // 1 tensor of shape [2, 3, 4]
std::iota(flat_data.begin(), flat_data.end(), 0.0);

primitive_array<double> values_array(flat_data);
fixed_shape_tensor_extension::metadata tensor_meta{
    {2, 3, 4},
    std::vector<std::string>{"X", "Y", "Z"},
    std::nullopt
};

std::vector<metadata_pair> arrow_metadata{
    {"author", "research_team"},
    {"version", "2.0"},
    {"experiment", "trial_42"}
};

const std::uint64_t list_size = tensor_meta.compute_size();

fixed_shape_tensor_array tensor_array(
    list_size,
    array(std::move(values_array)),
    tensor_meta,
    "neural_network_weights",  // array name
    arrow_metadata             // additional metadata
);

// Access the Arrow proxy to read metadata
const auto& proxy = tensor_array.get_arrow_proxy();
std::cout << "Array name: " << proxy.name() << "\n";

if (auto meta_opt = proxy.metadata())
{
    for (const auto& [key, value] : *meta_opt)
    {
        std::cout << key << ": " << value << "\n";
    }
}
```

### With Validity Bitmap

```cpp
#include "sparrow_extensions/fixed_shape_tensor.hpp"
using namespace sparrow_extensions;
using namespace sparrow;

// Create 4 tensors of shape [2, 2], with some null values
std::vector<int32_t> flat_data(16);
std::iota(flat_data.begin(), flat_data.end(), 0);

primitive_array<int32_t> values_array(flat_data);
fixed_shape_tensor_extension::metadata tensor_meta{{2, 2}, std::nullopt, std::nullopt};

// Create validity bitmap: first and third tensors are valid, others are null
std::vector<bool> validity{true, false, true, false};

const std::uint64_t list_size = tensor_meta.compute_size();

fixed_shape_tensor_array tensor_array(
    list_size,
    array(std::move(values_array)),
    tensor_meta,
    validity
);

// Check which tensors are valid
for (size_t i = 0; i < tensor_array.size(); ++i)
{
    auto tensor = tensor_array[i];
    if (tensor.has_value())
    {
        std::cout << "Tensor " << i << " is valid\n";
    }
    else
    {
        std::cout << "Tensor " << i << " is null\n";
    }
}
```

### JSON Metadata Serialization

```cpp
#include "sparrow_extensions/fixed_shape_tensor.hpp"
using namespace sparrow_extensions;

// Create metadata
fixed_shape_tensor_extension::metadata meta{
    {100, 200, 500},
    std::vector<std::string>{"C", "H", "W"},
    std::vector<std::int64_t>{2, 0, 1}
};

// Serialize to JSON
std::string json = meta.to_json();
// Result: {"shape":[100,200,500],"dim_names":["C","H","W"],"permutation":[2,0,1]}

// Deserialize from JSON
auto parsed_meta = fixed_shape_tensor_extension::metadata::from_json(json);

// Validate
if (parsed_meta.is_valid())
{
    std::cout << "Metadata is valid\n";
    std::cout << "Tensor size: " << parsed_meta.compute_size() << "\n";
}
```

Constructors
------------

The `fixed_shape_tensor_array` class provides five constructors to accommodate different use cases:

### 1. From Arrow Proxy (Reconstruction)

```cpp
fixed_shape_tensor_array(sparrow::arrow_proxy proxy);
```

Reconstructs a tensor array from an existing Arrow proxy. Used internally by the Arrow extension registry.

### 2. Basic Constructor

```cpp
template<typename T>
fixed_shape_tensor_array(
    std::uint64_t list_size,
    sparrow::array&& flat_values,
    const metadata_type& tensor_metadata
);
```

Creates a tensor array with the specified shape and flattened values. All tensors are valid (no null values).

**Parameters:**
- `list_size`: Product of all dimensions (from `tensor_metadata.compute_size()`)
- `flat_values`: Flattened tensor data as a primitive array
- `tensor_metadata`: Shape, optional dim_names, and optional permutation

### 3. With Name and Metadata

```cpp
template<typename T, sparrow::mpl::input_metadata_container M>
fixed_shape_tensor_array(
    std::uint64_t list_size,
    sparrow::array&& flat_values,
    const metadata_type& tensor_metadata,
    std::string_view name,
    M&& arrow_metadata = std::nullopt
);
```

Creates a tensor array with custom name and Arrow metadata fields.

**Parameters:**
- All parameters from basic constructor, plus:
- `name`: Name for the array (stored in Arrow schema)
- `arrow_metadata`: Optional additional metadata pairs

### 4. With Validity Bitmap

```cpp
template<typename T, sparrow::mpl::validity_bitmap_input VB>
fixed_shape_tensor_array(
    std::uint64_t list_size,
    sparrow::array&& flat_values,
    const metadata_type& tensor_metadata,
    VB&& validity
);
```

Creates a tensor array where some tensors may be null.

**Parameters:**
- All parameters from basic constructor, plus:
- `validity`: Validity bitmap (e.g., `std::vector<bool>`) indicating which tensors are valid

### 5. Complete Constructor

```cpp
template<typename T, sparrow::mpl::validity_bitmap_input VB, sparrow::mpl::input_metadata_container M>
fixed_shape_tensor_array(
    std::uint64_t list_size,
    sparrow::array&& flat_values,
    const metadata_type& tensor_metadata,
    VB&& validity,
    std::string_view name,
    M&& arrow_metadata = std::nullopt
);
```

Combines all features: validity bitmap, name, and metadata.

API Reference
-------------

### fixed_shape_tensor_array

The `fixed_shape_tensor_array` class wraps `sparrow::fixed_sized_list_array` with extension metadata for tensor storage.

| Method | Description |
| ------ | ----------- |
| `size() const` | Returns the number of tensors in the array |
| `shape() const` | Returns the shape vector for each tensor |
| `get_metadata() const` | Returns the complete tensor metadata (shape, dim_names, permutation) |
| `storage() const` | Returns const reference to underlying `fixed_sized_list_array` |
| `storage()` | Returns mutable reference to underlying `fixed_sized_list_array` |
| `operator[](size_type i) const` | Accesses the i-th tensor (returns nullable reference) |
| `get_arrow_proxy() const` | Returns const reference to Arrow proxy |
| `get_arrow_proxy()` | Returns mutable reference to Arrow proxy |

### fixed_shape_tensor_extension::metadata

The metadata structure for tensor arrays.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `shape` | `std::vector<std::int64_t>` | Dimensions of each tensor (required) |
| `dim_names` | `std::optional<std::vector<std::string>>` | Optional dimension names |
| `permutation` | `std::optional<std::vector<std::int64_t>>` | Optional dimension permutation |

| Method | Description |
| ------ | ----------- |
| `is_valid() const` | Validates metadata consistency |
| `compute_size() const` | Computes the product of all dimensions |
| `to_json() const` | Serializes metadata to JSON string |
| `static from_json(std::string_view)` | Deserializes metadata from JSON |

### Metadata Validation Rules

The `is_valid()` method checks:
- Shape is not empty and all dimensions are positive
- If `dim_names` is present, its size matches the shape size
- If `permutation` is present:
  - Its size matches the shape size
  - It contains exactly the indices [0, 1, ..., N-1] without duplicates
  - All indices are in valid range

Performance Considerations
--------------------------

### Optimizations

The implementation includes several performance optimizations:

1. **String reservation in JSON serialization**: Pre-calculates approximate JSON size to minimize allocations
2. **Direct string concatenation**: Avoids `std::ostringstream` overhead by using direct string operations
3. **Vector capacity hints**: Reserves space for typical tensor dimensions (2-4D) during JSON parsing
4. **Bitset-based permutation validation**: O(n) validation instead of O(n log n) sorting
5. **Move semantics**: Efficiently transfers metadata and arrays without copying
6. **Early returns**: Skips unnecessary work when extension metadata already exists

### Best Practices

- Use `compute_size()` to calculate the required `list_size` parameter
- Pre-allocate flat data vectors when possible
- Use move semantics when passing arrays to constructors
- Validate metadata with `is_valid()` before creating arrays
- For large tensors, consider memory layout and cache efficiency

Extension Metadata
------------------

The Fixed Shape Tensor array automatically sets the following Arrow extension metadata:

- `ARROW:extension:name`: `"arrow.fixed_shape_tensor"`
- `ARROW:extension:metadata`: JSON string containing shape, optional dim_names, and optional permutation

This metadata is added to the Arrow schema, allowing other Arrow implementations to recognize and correctly interpret the tensor arrays.

Examples from Specification
----------------------------

### Example 1: Simple 2×5 Tensor

```cpp
// From spec: { "shape": [2, 5] }
std::vector<double> data(10);
std::iota(data.begin(), data.end(), 0.0);

primitive_array<double> values(data);
fixed_shape_tensor_extension::metadata meta{{2, 5}, std::nullopt, std::nullopt};

fixed_shape_tensor_array tensors(
    meta.compute_size(),
    array(std::move(values)),
    meta
);
// Contains 1 tensor of shape [2, 5]
```

### Example 2: Image Data with Dimension Names

```cpp
// From spec: { "shape": [100, 200, 500], "dim_names": ["C", "H", "W"] }
std::vector<float> image_data(100 * 200 * 500);
// Fill with image data...

primitive_array<float> values(image_data);
fixed_shape_tensor_extension::metadata meta{
    {100, 200, 500},
    std::vector<std::string>{"C", "H", "W"},
    std::nullopt
};

fixed_shape_tensor_array images(
    meta.compute_size(),
    array(std::move(values)),
    meta
);
// Contains 1 tensor representing channels × height × width
```

### Example 3: Permuted Layout

```cpp
// From spec: { "shape": [100, 200, 500], "permutation": [2, 0, 1] }
// Physical layout: [100, 200, 500]
// Logical layout: [500, 100, 200]

std::vector<float> data(100 * 200 * 500);
// Fill with data in physical layout...

primitive_array<float> values(data);
fixed_shape_tensor_extension::metadata meta{
    {100, 200, 500},
    std::nullopt,
    std::vector<std::int64_t>{2, 0, 1}
};

fixed_shape_tensor_array tensors(
    meta.compute_size(),
    array(std::move(values)),
    meta
);
// Data is stored in physical layout but can be interpreted with logical shape
```

See Also
--------

- [Apache Arrow Canonical Extension: FixedShapeTensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor)
- [sparrow::fixed_sized_list_array](https://github.com/man-group/sparrow)
- [Bool8 Array](@ref bool8_array)
- [UUID Array](@ref uuid_array)
- [JSON Array](@ref json_array)
