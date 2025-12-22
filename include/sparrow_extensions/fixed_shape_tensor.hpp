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

#include "sparrow/array.hpp"
#include "sparrow/list_array.hpp"
#include "sparrow/types/data_type.hpp"
#include "sparrow/utils/contracts.hpp"

namespace sparrow_extensions
{
    /**
     * @brief Fixed shape tensor array implementation following Arrow canonical extension
     * specification.
     *
     * This class implements an Arrow-compatible array for storing fixed-shape tensors
     * according to the Apache Arrow canonical extension specification for fixed shape tensors.
     * Each tensor is stored as a FixedSizeList with a product of shape dimensions as list size.
     *
     * The fixed shape tensor extension type is defined as:
     * - Extension name: "arrow.fixed_shape_tensor"
     * - Storage type: FixedSizeList where:
     *   - value_type is the data type of individual tensor elements
     *   - list_size is the product of all elements in tensor shape
     *
     * Extension type parameters:
     * - value_type: the Arrow data type of individual tensor elements
     * - shape: the physical shape of the contained tensors as an array
     * - dim_names (optional): explicit names to tensor dimensions as an array
     * - permutation (optional): indices of the desired ordering of the original dimensions
     *
     * The metadata must be a valid JSON object including shape of the contained tensors
     * as an array with key "shape" plus optional dimension names with keys "dim_names"
     * and ordering of the dimensions with key "permutation".
     *
     * Example metadata:
     * - Simple shape: { "shape": [2, 5] }
     * - With dim_names: { "shape": [100, 200, 500], "dim_names": ["C", "H", "W"] }
     * - With permutation: { "shape": [100, 200, 500], "permutation": [2, 0, 1] }
     *
     * Note: Elements in a fixed shape tensor extension array are stored in
     * row-major/C-contiguous order.
     *
     * Related Apache Arrow specification:
     * https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor
     */
    class fixed_shape_tensor_extension
    {
    public:

        static constexpr std::string_view EXTENSION_NAME = "arrow.fixed_shape_tensor";

        /**
         * @brief Metadata for fixed shape tensor extension.
         *
         * Stores the shape, optional dimension names, and optional permutation
         * for the tensor layout.
         */
        struct metadata
        {
            std::vector<std::int64_t> shape;
            std::optional<std::vector<std::string>> dim_names;
            std::optional<std::vector<std::int64_t>> permutation;

            /**
             * @brief Validates that the metadata is well-formed.
             *
             * @return true if metadata is valid, false otherwise
             *
             * Validation rules:
             * - shape must not be empty
             * - shape elements must all be positive
             * - if dim_names is present, its size must equal shape size
             * - if permutation is present:
             *   - its size must equal shape size
             *   - it must contain exactly the values [0, 1, ..., N-1] in some order
             */
            [[nodiscard]] bool is_valid() const;

            /**
             * @brief Computes the total number of elements (product of shape).
             *
             * @return Product of all dimensions in shape
             */
            [[nodiscard]] std::int64_t compute_size() const;

            /**
             * @brief Serializes metadata to JSON string.
             *
             * @return JSON string representation of the metadata
             */
            [[nodiscard]] std::string to_json() const;

            /**
             * @brief Deserializes metadata from JSON string.
             *
             * @param json JSON string to parse
             * @return Parsed metadata structure
             * @throws std::runtime_error if JSON is invalid
             */
            [[nodiscard]] static metadata from_json(std::string_view json);
        };

        /**
         * @brief Initializes the extension metadata on an arrow proxy.
         *
         * @param proxy Arrow proxy to initialize
         * @param tensor_metadata Metadata describing the tensor shape and layout
         *
         * @pre proxy must represent a FixedSizeList
         * @pre tensor_metadata must be valid
         * @post Extension metadata is added to the proxy
         */
        static void init(sparrow::arrow_proxy& proxy, const metadata& tensor_metadata);

        /**
         * @brief Extracts metadata from an arrow proxy.
         *
         * @param proxy Arrow proxy to extract metadata from
         * @return Metadata structure parsed from the proxy's extension metadata
         * @throws std::runtime_error if metadata is missing or invalid
         */
        [[nodiscard]] static metadata extract_metadata(const sparrow::arrow_proxy& proxy);
    };

    /**
     * @brief Fixed shape tensor array wrapping a fixed_sized_list_array.
     *
     * This class provides a convenient interface for working with fixed-shape tensors
     * while maintaining compatibility with the Arrow format.
     */
    class fixed_shape_tensor_array
    {
    public:

        using size_type = std::size_t;
        using metadata_type = fixed_shape_tensor_extension::metadata;

        /**
         * @brief Constructs a fixed shape tensor array from an arrow proxy.
         *
         * @param proxy Arrow proxy containing the tensor data
         *
         * @pre proxy must contain valid Fixed Size List array data
         * @pre proxy must have valid extension metadata
         * @post Array is initialized with data from proxy
         */
        explicit fixed_shape_tensor_array(sparrow::arrow_proxy proxy);

        /**
         * @brief Constructs a fixed shape tensor array from values and shape.
         *
         * @param list_size Total number of elements per tensor (product of shape)
         * @param flat_values Flattened sparrow array of all tensor elements in row-major order
         * @param tensor_metadata Metadata describing the tensor shape and layout
         * @param nullable Whether the array should support null values
         *
         * @pre flat_values.size() must be divisible by list_size
         * @pre list_size must equal tensor_metadata.compute_size()
         * @pre tensor_metadata must be valid
         * @post Array contains tensors reshaped according to the metadata
         */
        fixed_shape_tensor_array(
            std::uint64_t list_size,
            sparrow::array&& flat_values,
            const metadata_type& tensor_metadata,
            bool nullable = true
        );

        // Default special members
        fixed_shape_tensor_array(const fixed_shape_tensor_array&) = default;
        fixed_shape_tensor_array& operator=(const fixed_shape_tensor_array&) = default;
        fixed_shape_tensor_array(fixed_shape_tensor_array&&) noexcept = default;
        fixed_shape_tensor_array& operator=(fixed_shape_tensor_array&&) noexcept = default;
        ~fixed_shape_tensor_array() = default;

        /**
         * @brief Returns the number of tensors in the array.
         */
        [[nodiscard]] size_type size() const;

        /**
         * @brief Returns the metadata describing the tensor shape and layout.
         */
        [[nodiscard]] const metadata_type& get_metadata() const;

        /**
         * @brief Returns the shape of each tensor.
         */
        [[nodiscard]] const std::vector<std::int64_t>& shape() const;

        /**
         * @brief Returns the underlying fixed_sized_list_array.
         */
        [[nodiscard]] const sparrow::fixed_sized_list_array& storage() const;

        /**
         * @brief Returns the underlying fixed_sized_list_array.
         */
        [[nodiscard]] sparrow::fixed_sized_list_array& storage();

        /**
         * @brief Access tensor at index i.
         *
         * @param i Index of the tensor
         * @return A list_value representing the tensor at index i
         *
         * @pre i < size()
         */
        [[nodiscard]] auto operator[](size_type i) const -> decltype(std::declval<const sparrow::fixed_sized_list_array&>()[i]);

        /**
         * @brief Returns the underlying arrow_proxy.
         */
        [[nodiscard]] const sparrow::arrow_proxy& get_arrow_proxy() const;

        /**
         * @brief Returns the underlying arrow_proxy.
         */
        [[nodiscard]] sparrow::arrow_proxy& get_arrow_proxy();

    private:

        sparrow::fixed_sized_list_array m_storage;
        metadata_type m_metadata;
    };

}  // namespace sparrow_extensions

namespace sparrow::detail
{
    template <>
    struct get_data_type_from_array<sparrow_extensions::fixed_shape_tensor_array>
    {
        [[nodiscard]] static constexpr sparrow::data_type get()
        {
            return sparrow::data_type::FIXED_SIZED_LIST;
        }
    };
}
