#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sparrow/buffer/dynamic_bitset/dynamic_bitset.hpp"
#include "sparrow/list_array.hpp"
#include "sparrow/types/data_type.hpp"

#include "sparrow_extensions/config/config.hpp"

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
        struct SPARROW_EXTENSIONS_API metadata
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
    class SPARROW_EXTENSIONS_API fixed_shape_tensor_array
    {
    public:

        using size_type = std::size_t;
        using metadata_type = fixed_shape_tensor_extension::metadata;

        using inner_value_type = sparrow::fixed_sized_list_array::inner_value_type;
        using inner_reference = sparrow::fixed_sized_list_array::inner_reference;
        using inner_const_reference = sparrow::fixed_sized_list_array::inner_const_reference;

        using bitmap_type = sparrow::fixed_sized_list_array::bitmap_type;
        using bitmap_const_reference = sparrow::fixed_sized_list_array::bitmap_const_reference;

        using value_type = sparrow::nullable<inner_value_type>;
        using const_reference = sparrow::nullable<inner_const_reference, bitmap_const_reference>;

        using const_iterator = sparrow::fixed_sized_list_array::const_iterator;
        using const_reverse_iterator = sparrow::fixed_sized_list_array::const_reverse_iterator;

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
         *
         * @pre flat_values.size() must be divisible by list_size
         * @pre list_size must equal tensor_metadata.compute_size()
         * @pre tensor_metadata must be valid
         * @post Array contains tensors reshaped according to the metadata
         */
        fixed_shape_tensor_array(
            std::uint64_t list_size,
            sparrow::array&& flat_values,
            const metadata_type& tensor_metadata
        );

        /**
         * @brief Constructs a fixed shape tensor array with name and/or metadata.
         *
         * @param list_size Total number of elements per tensor (product of shape)
         * @param flat_values Flattened sparrow array of all tensor elements in row-major order
         * @param tensor_metadata Metadata describing the tensor shape and layout
         * @param name Name for the array
         * @param arrow_metadata Optional Arrow metadata key-value pairs
         *
         * @pre flat_values.size() must be divisible by list_size
         * @pre list_size must equal tensor_metadata.compute_size()
         * @pre tensor_metadata must be valid
         * @post Array contains tensors with the specified name and metadata
         */
        fixed_shape_tensor_array(
            std::uint64_t list_size,
            sparrow::array&& flat_values,
            const metadata_type& tensor_metadata,
            std::string_view name,
            std::optional<std::vector<sparrow::metadata_pair>> arrow_metadata = std::nullopt
        );

        /**
         * @brief Constructs a fixed shape tensor array with validity bitmap.
         *
         * @tparam VB Type of validity bitmap input
         * @param list_size Total number of elements per tensor (product of shape)
         * @param flat_values Flattened sparrow array of all tensor elements in row-major order
         * @param tensor_metadata Metadata describing the tensor shape and layout
         * @param validity_input Validity bitmap (one bit per tensor)
         *
         * @pre flat_values.size() must be divisible by list_size
         * @pre list_size must equal tensor_metadata.compute_size()
         * @pre tensor_metadata must be valid
         * @pre validity_input size must match number of tensors (flat_values.size() / list_size)
         * @post Array contains tensors with the specified validity bitmap
         */
        template <sparrow::validity_bitmap_input VB>
        fixed_shape_tensor_array(
            std::uint64_t list_size,
            sparrow::array&& flat_values,
            const metadata_type& tensor_metadata,
            VB&& validity_input
        );

        /**
         * @brief Constructs a fixed shape tensor array with validity, name, and metadata.
         *
         * @tparam VB Type of validity bitmap input
         * @tparam METADATA_RANGE Type of metadata container
         * @param list_size Total number of elements per tensor (product of shape)
         * @param flat_values Flattened sparrow array of all tensor elements in row-major order
         * @param tensor_metadata Metadata describing the tensor shape and layout
         * @param validity_input Validity bitmap (one bit per tensor)
         * @param name Optional name for the array
         * @param arrow_metadata Optional Arrow metadata key-value pairs
         *
         * @pre flat_values.size() must be divisible by list_size
         * @pre list_size must equal tensor_metadata.compute_size()
         * @pre tensor_metadata must be valid
         * @pre validity_input size must match number of tensors (flat_values.size() / list_size)
         * @post Array contains tensors with the specified validity bitmap, name, and metadata
         */
        template <
            sparrow::validity_bitmap_input VB,
            sparrow::input_metadata_container METADATA_RANGE = std::vector<sparrow::metadata_pair>>
        fixed_shape_tensor_array(
            std::uint64_t list_size,
            sparrow::array&& flat_values,
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
        [[nodiscard]] const_reference operator[](size_type i) const;

        /**
         * @brief Returns the underlying arrow_proxy.
         */
        [[nodiscard]] const sparrow::arrow_proxy& get_arrow_proxy() const;

        /**
         * @brief Returns the underlying arrow_proxy.
         */
        [[nodiscard]] sparrow::arrow_proxy& get_arrow_proxy();

        /**
         * @brief Checks if the array is empty.
         *
         * @return true if size() == 0, false otherwise
         */
        [[nodiscard]] bool empty() const;

        /**
         * @brief Bounds-checked access to tensor at index i.
         *
         * @param i Index of the tensor
         * @return A list_value representing the tensor at index i
         * @throws std::out_of_range if i >= size()
         */
        [[nodiscard]] const_reference at(size_type i) const;

        /**
         * @brief Validates that the array structure is well-formed.
         *
         * @return true if metadata is valid, false otherwise
         */
        [[nodiscard]] bool is_valid() const;

        /**
         * @brief Returns the validity bitmap.
         *
         * @return Range representing the validity bitmap
         */
        [[nodiscard]] auto bitmap() const
        {
            return m_storage.bitmap();
        }

        /**
         * @brief Returns iterator to the beginning.
         */
        [[nodiscard]] const_iterator begin() const;

        /**
         * @brief Returns iterator to the end.
         */
        [[nodiscard]] const_iterator end() const;

        /**
         * @brief Returns const iterator to the beginning.
         */
        [[nodiscard]] const_iterator cbegin() const;

        /**
         * @brief Returns const iterator to the end.
         */
        [[nodiscard]] const_iterator cend() const;

        /**
         * @brief Returns reverse iterator to the beginning.
         */
        [[nodiscard]] const_reverse_iterator rbegin() const;

        /**
         * @brief Returns reverse iterator to the end.
         */
        [[nodiscard]] const_reverse_iterator rend() const;

        /**
         * @brief Returns const reverse iterator to the beginning.
         */
        [[nodiscard]] const_reverse_iterator crbegin() const;

        /**
         * @brief Returns const reverse iterator to the end.
         */
        [[nodiscard]] const_reverse_iterator crend() const;

    private:

        void finalize_construction();

        sparrow::fixed_sized_list_array m_storage;
        metadata_type m_metadata;
    };

    // Template constructor implementations

    template <sparrow::validity_bitmap_input VB>
    fixed_shape_tensor_array::fixed_shape_tensor_array(
        std::uint64_t list_size,
        sparrow::array&& flat_values,
        const metadata_type& tensor_metadata,
        VB&& validity_input
    )
        : m_storage(list_size, std::move(flat_values), std::forward<VB>(validity_input))
        , m_metadata(tensor_metadata)
    {
        SPARROW_ASSERT_TRUE(m_metadata.is_valid());
        SPARROW_ASSERT_TRUE(static_cast<std::int64_t>(list_size) == m_metadata.compute_size());

        finalize_construction();
    }

    template <sparrow::validity_bitmap_input VB, sparrow::input_metadata_container METADATA_RANGE>
    fixed_shape_tensor_array::fixed_shape_tensor_array(
        std::uint64_t list_size,
        sparrow::array&& flat_values,
        const metadata_type& tensor_metadata,
        VB&& validity_input,
        std::optional<std::string_view> name,
        std::optional<METADATA_RANGE> arrow_metadata
    )
        : m_storage(list_size, std::move(flat_values), std::forward<VB>(validity_input))
        , m_metadata(tensor_metadata)
    {
        SPARROW_ASSERT_TRUE(m_metadata.is_valid());
        SPARROW_ASSERT_TRUE(static_cast<std::int64_t>(list_size) == m_metadata.compute_size());

        // Get the proxy and set name/metadata if provided
        auto& proxy = sparrow::detail::array_access::get_arrow_proxy(m_storage);

        if (name.has_value())
        {
            proxy.set_name(*name);
        }

        if (arrow_metadata.has_value())
        {
            proxy.set_metadata(std::make_optional(*arrow_metadata));
        }

        finalize_construction();
    }

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
