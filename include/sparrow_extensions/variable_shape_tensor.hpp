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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sparrow/buffer/dynamic_bitset/dynamic_bitset.hpp"
#include "sparrow/struct_array.hpp"
#include "sparrow/types/data_type.hpp"

#include "sparrow_extensions/config/config.hpp"

namespace sparrow_extensions
{
    /**
     * @brief Variable shape tensor array implementation following Arrow canonical extension
     * specification.
     *
     * This class implements an Arrow-compatible array for storing variable-shape tensors
     * according to the Apache Arrow canonical extension specification for variable shape tensors.
     * Each tensor can have a different shape, and is stored in a StructArray with data and shape fields.
     *
     * The variable shape tensor extension type is defined as:
     * - Extension name: "arrow.variable_shape_tensor"
     * - Storage type: StructArray where struct is composed of:
     *   - data: List holding tensor elements (each list element is a single tensor)
     *   - shape: FixedSizeList<int32>[ndim] of the tensor shape
     *
     * Extension type parameters:
     * - value_type: the Arrow data type of individual tensor elements
     *
     * Optional parameters describing the logical layout:
     * - dim_names: explicit names to tensor dimensions as an array
     * - permutation: indices of the desired ordering of the original dimensions
     * - uniform_shape: sizes of individual tensor's dimensions which are guaranteed to stay
     *   constant in uniform dimensions (represented with int32 values) and can vary in
     *   non-uniform dimensions (represented with null)
     *
     * Example metadata:
     * - With dim_names: { "dim_names": ["C", "H", "W"] }
     * - With uniform_shape: { "dim_names": ["H", "W", "C"], "uniform_shape": [400, null, 3] }
     * - With permutation: { "permutation": [2, 0, 1] }
     *
     * Note: Values inside each data tensor element are stored in row-major/C-contiguous order
     * according to the corresponding shape.
     *
     * Related Apache Arrow specification:
     * https://arrow.apache.org/docs/format/CanonicalExtensions.html#variable-shape-tensor
     */
    class variable_shape_tensor_extension
    {
    public:

        static constexpr std::string_view EXTENSION_NAME = "arrow.variable_shape_tensor";

        /**
         * @brief Metadata for variable shape tensor extension.
         *
         * Stores optional dimension names, permutation, and uniform shape information
         * for the tensor layout.
         */
        struct SPARROW_EXTENSIONS_API metadata
        {
            std::optional<std::vector<std::string>> dim_names;
            std::optional<std::vector<std::int64_t>> permutation;
            std::optional<std::vector<std::optional<std::int32_t>>> uniform_shape;

            /**
             * @brief Validates that the metadata is well-formed.
             *
             * @return true if metadata is valid, false otherwise
             *
             * Validation rules:
             * - if dim_names, permutation, and uniform_shape are all present, they must have the same size
             * - if permutation is present, it must contain exactly the values [0, 1, ..., N-1] in some order
             * - if uniform_shape is present and contains non-null values, they must all be positive
             */
            [[nodiscard]] bool is_valid() const;

            /**
             * @brief Gets the number of dimensions if it can be determined from metadata.
             *
             * @return Number of dimensions if determinable, nullopt otherwise
             *
             * The number of dimensions can be determined if any of dim_names, permutation,
             * or uniform_shape is present.
             */
            [[nodiscard]] std::optional<std::size_t> get_ndim() const;

            /**
             * @brief Serializes metadata to JSON string.
             *
             * @return JSON string representation of the metadata (may be empty "{}")
             */
            [[nodiscard]] std::string to_json() const;

            /**
             * @brief Deserializes metadata from JSON string.
             *
             * @param json JSON string to parse (may be empty or "{}")
             * @return Parsed metadata structure
             * @throws std::runtime_error if JSON is invalid
             */
            [[nodiscard]] static metadata from_json(std::string_view json);
        };

        /**
         * @brief Initializes the extension metadata on an arrow proxy.
         *
         * @param proxy Arrow proxy to initialize
         * @param tensor_metadata Metadata describing the tensor layout
         *
         * @pre proxy must represent a StructArray with data and shape fields
         * @pre tensor_metadata must be valid
         * @post Extension metadata is added to the proxy
         */
        static void init(sparrow::arrow_proxy& proxy, const metadata& tensor_metadata);

        /**
         * @brief Extracts metadata from an arrow proxy.
         *
         * @param proxy Arrow proxy to extract metadata from
         * @return Metadata structure parsed from the proxy's extension metadata
         * @throws std::runtime_error if metadata format is invalid
         *
         * Note: Returns default metadata if no extension metadata is present
         */
        [[nodiscard]] static metadata extract_metadata(const sparrow::arrow_proxy& proxy);
    };

    /**
     * @brief Variable shape tensor array wrapping a struct_array.
     *
     * This class provides a convenient interface for working with variable-shape tensors
     * while maintaining compatibility with the Arrow format. Each tensor can have a different
     * shape, and the shapes are stored alongside the data.
     */
    class SPARROW_EXTENSIONS_API variable_shape_tensor_array
    {
    public:

        using size_type = std::size_t;
        using metadata_type = variable_shape_tensor_extension::metadata;

        // Typedefs inherited from struct_array
        using inner_value_type = sparrow::struct_value;
        using inner_reference = sparrow::struct_value;
        using inner_const_reference = sparrow::struct_value;

        using bitmap_type = sparrow::struct_array::bitmap_type;
        using bitmap_const_reference = sparrow::struct_array::bitmap_const_reference;

        using value_type = sparrow::nullable<inner_value_type>;
        using const_reference = sparrow::nullable<inner_const_reference, bitmap_const_reference>;

        using const_iterator = sparrow::struct_array::const_iterator;
        using const_reverse_iterator = sparrow::struct_array::const_reverse_iterator;

        /**
         * @brief Constructs a variable shape tensor array from an arrow proxy.
         *
         * @param proxy Arrow proxy containing the tensor data
         *
         * @pre proxy must contain valid StructArray data with data and shape fields
         * @pre proxy must have valid extension metadata
         * @post Array is initialized with data from proxy
         */
        explicit variable_shape_tensor_array(sparrow::arrow_proxy proxy);

        /**
         * @brief Constructs a variable shape tensor array from data and shapes.
         *
         * @param ndim Number of dimensions for all tensors
         * @param tensor_data List array containing flattened tensor data (one list per tensor)
         * @param tensor_shapes FixedSizeList array containing shapes (one shape per tensor)
         * @param tensor_metadata Metadata describing the tensor layout
         *
         * @pre tensor_data.size() must equal tensor_shapes.size()
         * @pre tensor_shapes list_size must equal ndim
         * @pre tensor_metadata must be valid
         * @post Array contains tensors with the specified data and shapes
         */
        variable_shape_tensor_array(
            std::uint64_t ndim,
            sparrow::array&& tensor_data,
            sparrow::array&& tensor_shapes,
            const metadata_type& tensor_metadata
        );

        /**
         * @brief Constructs a variable shape tensor array with name and/or metadata.
         *
         * @param ndim Number of dimensions for all tensors
         * @param tensor_data List array containing flattened tensor data (one list per tensor)
         * @param tensor_shapes FixedSizeList array containing shapes (one shape per tensor)
         * @param tensor_metadata Metadata describing the tensor layout
         * @param name Name for the array
         * @param arrow_metadata Optional Arrow metadata key-value pairs
         *
         * @pre tensor_data.size() must equal tensor_shapes.size()
         * @pre tensor_shapes list_size must equal ndim
         * @pre tensor_metadata must be valid
         * @post Array contains tensors with the specified name and metadata
         */
        variable_shape_tensor_array(
            std::uint64_t ndim,
            sparrow::array&& tensor_data,
            sparrow::array&& tensor_shapes,
            const metadata_type& tensor_metadata,
            std::string_view name,
            std::optional<std::vector<sparrow::metadata_pair>> arrow_metadata = std::nullopt
        );

        /**
         * @brief Constructs a variable shape tensor array with validity bitmap.
         *
         * @tparam VB Type of validity bitmap input
         * @param ndim Number of dimensions for all tensors
         * @param tensor_data List array containing flattened tensor data (one list per tensor)
         * @param tensor_shapes FixedSizeList array containing shapes (one shape per tensor)
         * @param tensor_metadata Metadata describing the tensor layout
         * @param validity_input Validity bitmap (one bit per tensor)
         *
         * @pre tensor_data.size() must equal tensor_shapes.size()
         * @pre tensor_shapes list_size must equal ndim
         * @pre tensor_metadata must be valid
         * @pre validity_input size must match number of tensors
         * @post Array contains tensors with the specified validity bitmap
         */
        template <sparrow::validity_bitmap_input VB>
        variable_shape_tensor_array(
            std::uint64_t ndim,
            sparrow::array&& tensor_data,
            sparrow::array&& tensor_shapes,
            const metadata_type& tensor_metadata,
            VB&& validity_input
        );

        /**
         * @brief Constructs a variable shape tensor array with validity, name, and metadata.
         *
         * @tparam VB Type of validity bitmap input
         * @tparam METADATA_RANGE Type of metadata container
         * @param ndim Number of dimensions for all tensors
         * @param tensor_data List array containing flattened tensor data (one list per tensor)
         * @param tensor_shapes FixedSizeList array containing shapes (one shape per tensor)
         * @param tensor_metadata Metadata describing the tensor layout
         * @param validity_input Validity bitmap (one bit per tensor)
         * @param name Optional name for the array
         * @param arrow_metadata Optional Arrow metadata key-value pairs
         *
         * @pre tensor_data.size() must equal tensor_shapes.size()
         * @pre tensor_shapes list_size must equal ndim
         * @pre tensor_metadata must be valid
         * @pre validity_input size must match number of tensors
         * @post Array contains tensors with the specified validity bitmap, name, and metadata
         */
        template <sparrow::validity_bitmap_input VB, sparrow::input_metadata_container METADATA_RANGE = std::vector<sparrow::metadata_pair>>
        variable_shape_tensor_array(
            std::uint64_t ndim,
            sparrow::array&& tensor_data,
            sparrow::array&& tensor_shapes,
            const metadata_type& tensor_metadata,
            VB&& validity_input,
            std::optional<std::string_view> name,
            std::optional<METADATA_RANGE> arrow_metadata = std::nullopt
        );

        /**
         * @brief Returns the number of tensors in the array.
         */
        [[nodiscard]] size_type size() const;

        /**
         * @brief Returns the metadata describing the tensor layout.
         */
        [[nodiscard]] const metadata_type& get_metadata() const;

        /**
         * @brief Returns the number of dimensions if it can be determined.
         *
         * @return Number of dimensions if determinable, nullopt otherwise
         */
        [[nodiscard]] std::optional<std::size_t> ndim() const;

        /**
         * @brief Returns the underlying struct_array.
         */
        [[nodiscard]] const sparrow::struct_array& storage() const;

        /**
         * @brief Returns the underlying struct_array.
         */
        [[nodiscard]] sparrow::struct_array& storage();

        /**
         * @brief Access tensor at index i.
         *
         * @param i Index of the tensor
         * @return A struct_value representing the tensor at index i
         *
         * @pre i < size()
         */
        [[nodiscard]] const_reference operator[](size_type i) const
        {
            return m_storage[i];
        }

        /**
         * @brief Returns the underlying arrow_proxy.
         */
        [[nodiscard]] const sparrow::arrow_proxy& get_arrow_proxy() const;

        /**
         * @brief Returns the underlying arrow_proxy.
         */
        [[nodiscard]] sparrow::arrow_proxy& get_arrow_proxy();

        /**
         * @brief Gets const pointer to the data child array.
         *
         * @return Const pointer to the data array wrapper (index 0)
         */
        [[nodiscard]] const sparrow::array_wrapper* data_child() const;

        /**
         * @brief Gets mutable pointer to the data child array.
         *
         * @return Pointer to the data array wrapper (index 0)
         */
        [[nodiscard]] sparrow::array_wrapper* data_child();

        /**
         * @brief Gets const pointer to the shape child array.
         *
         * @return Const pointer to the shape array wrapper (index 1)
         */
        [[nodiscard]] const sparrow::array_wrapper* shape_child() const;

        /**
         * @brief Gets mutable pointer to the shape child array.
         *
         * @return Pointer to the shape array wrapper (index 1)
         */
        [[nodiscard]] sparrow::array_wrapper* shape_child();

        /**
         * @brief Gets the names of all child arrays.
         *
         * @return Range of child array names
         */
        [[nodiscard]] auto names() const
        {
            return m_storage.names();
        }

        /**
         * @brief Gets the const bitmap range for iteration.
         *
         * @return Const bitmap range
         */
        [[nodiscard]] auto bitmap() const
        {
            return m_storage.bitmap();
        }

        /**
         * @brief Checks if the array is empty.
         *
         * @return true if size() == 0, false otherwise
         */
        [[nodiscard]] bool empty() const
        {
            return size() == 0;
        }

        /**
         * @brief Access tensor at index i with bounds checking.
         *
         * @param i Index of the tensor
         * @return A struct_value representing the tensor at index i
         *
         * @throws std::out_of_range if i >= size()
         */
        [[nodiscard]] const_reference at(size_type i) const;

        /**
         * @brief Validates the internal structure of the tensor array.
         *
         * @return true if structure is valid (2 children, metadata valid), false otherwise
         */
        [[nodiscard]] bool is_valid() const;

        /**
         * @brief Returns the name of the data field.
         *
         * @return "data"
         */
        [[nodiscard]] static constexpr std::string_view data_field_name()
        {
            return "data";
        }

        /**
         * @brief Returns the name of the shape field.
         *
         * @return "shape"
         */
        [[nodiscard]] static constexpr std::string_view shape_field_name()
        {
            return "shape";
        }

        /**
         * @brief Returns iterator to the beginning of the tensor array.
         *
         * @return Iterator to the first tensor
         */
        [[nodiscard]] const_iterator begin() const;

        /**
         * @brief Returns iterator to the end of the tensor array.
         *
         * @return Iterator past the last tensor
         */
        [[nodiscard]] const_iterator end() const;

        /**
         * @brief Returns const iterator to the beginning of the tensor array.
         *
         * @return Const iterator to the first tensor
         */
        [[nodiscard]] const_iterator cbegin() const;
        /**
         * @brief Returns const iterator to the end of the tensor array.
         *
         * @return Const iterator past the last tensor
         */
        [[nodiscard]] const_iterator cend() const;

        /**
         * @brief Returns reverse iterator to the beginning of the reversed tensor array.
         *
         * @return Reverse iterator to the first tensor in reverse order
         */
        [[nodiscard]] const_reverse_iterator rbegin() const;

        /**
         * @brief Returns reverse iterator to the end of the reversed tensor array.
         *
         * @return Reverse iterator past the last tensor in reverse order
         */
        [[nodiscard]] const_reverse_iterator rend() const;

        /**
         * @brief Returns const reverse iterator to the beginning of the reversed tensor array.
         *
         * @return Const reverse iterator to the first tensor in reverse order
         */
        [[nodiscard]] const_reverse_iterator crbegin() const;
        /**
         * @brief Returns const reverse iterator to the end of the reversed tensor array.
         *
         * @return Const reverse iterator past the last tensor in reverse order
         */
        [[nodiscard]] const_reverse_iterator crend() const;

    private:

        void validate_and_init(
            std::uint64_t ndim,
            std::optional<std::string_view> name = std::nullopt,
            std::optional<std::vector<sparrow::metadata_pair>>* arrow_metadata = nullptr
        );

        sparrow::struct_array m_storage;
        metadata_type m_metadata;
    };

    // Helper function to construct the struct array with named fields
    namespace detail
    {
        template <typename VB = bool>
        sparrow::struct_array make_tensor_struct(
            sparrow::array&& tensor_data,
            sparrow::array&& tensor_shapes,
            VB&& validity_input = false
        )
        {
            // Set names on the arrays
            sparrow::detail::array_access::get_arrow_proxy(tensor_data).set_name("data");
            sparrow::detail::array_access::get_arrow_proxy(tensor_shapes).set_name("shape");
            
            // Construct the struct array
            std::vector<sparrow::array> children;
            children.push_back(std::move(tensor_data));
            children.push_back(std::move(tensor_shapes));
            return sparrow::struct_array(std::move(children), std::forward<VB>(validity_input));
        }
    }

    // Template constructor implementations

    template <sparrow::validity_bitmap_input VB>
    variable_shape_tensor_array::variable_shape_tensor_array(
        std::uint64_t ndim,
        sparrow::array&& tensor_data,
        sparrow::array&& tensor_shapes,
        const metadata_type& tensor_metadata,
        VB&& validity_input
    )
        : m_storage(detail::make_tensor_struct(std::move(tensor_data), std::move(tensor_shapes), std::forward<VB>(validity_input)))
        , m_metadata(tensor_metadata)
    {
        validate_and_init(ndim);
    }

    template <sparrow::validity_bitmap_input VB, sparrow::input_metadata_container METADATA_RANGE>
    variable_shape_tensor_array::variable_shape_tensor_array(
        std::uint64_t ndim,
        sparrow::array&& tensor_data,
        sparrow::array&& tensor_shapes,
        const metadata_type& tensor_metadata,
        VB&& validity_input,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> arrow_metadata
    )
        : m_storage(detail::make_tensor_struct(std::move(tensor_data), std::move(tensor_shapes), std::forward<VB>(validity_input)))
        , m_metadata(tensor_metadata)
    {
        std::optional<std::vector<sparrow::metadata_pair>> metadata_opt;
        if (arrow_metadata.has_value())
        {
            metadata_opt = std::vector<sparrow::metadata_pair>(arrow_metadata->begin(), arrow_metadata->end());
        }
        validate_and_init(ndim, name, arrow_metadata.has_value() ? &metadata_opt : nullptr);
    }

}  // namespace sparrow_extensions

namespace sparrow::detail
{
    template <>
    struct get_data_type_from_array<sparrow_extensions::variable_shape_tensor_array>
    {
        [[nodiscard]] static constexpr sparrow::data_type get()
        {
            return sparrow::data_type::STRUCT;
        }
    };
}
