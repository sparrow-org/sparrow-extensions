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

#include "sparrow_extensions/variable_shape_tensor.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>

#include <simdjson.h>

#include "sparrow/array.hpp"
#include "sparrow/layout/array_access.hpp"
#include "sparrow/layout/array_registry.hpp"
#include "sparrow/utils/contracts.hpp"

#include "sparrow_extensions/config/config.hpp"

namespace sparrow_extensions
{
    namespace
    {
        // JSON serialization size estimation constants
        constexpr std::size_t json_base_size = 2;              // {}
        constexpr std::size_t json_integer_avg_size = 10;      // Average size per integer
        constexpr std::size_t json_dim_names_overhead = 15;    // ,"dim_names":[]
        constexpr std::size_t json_string_overhead = 3;        // "name",
        constexpr std::size_t json_permutation_overhead = 17;  // ,"permutation":[]
        constexpr std::size_t json_uniform_shape_overhead = 18;  // ,"uniform_shape":[]
        constexpr std::size_t json_null_size = 4;              // null

        // JSON parsing capacity hints
        constexpr std::size_t typical_tensor_dimensions = 8;  // Typical tensor rank (2-4 dims, reserve 8)
    }

    // Metadata implementation
    
    std::optional<std::size_t> variable_shape_tensor_extension::metadata::get_ndim() const
    {
        if (dim_names.has_value())
        {
            return dim_names->size();
        }
        if (permutation.has_value())
        {
            return permutation->size();
        }
        if (uniform_shape.has_value())
        {
            return uniform_shape->size();
        }
        return std::nullopt;
    }

    bool variable_shape_tensor_extension::metadata::is_valid() const
    {
        // Determine the expected dimension count from the first available source
        const auto expected_ndim = get_ndim();

        // If we have an expected dimension, validate all present arrays match it
        if (expected_ndim.has_value())
        {
            const auto ndim = *expected_ndim;
            if ((dim_names.has_value() && dim_names->size() != ndim)
                || (permutation.has_value() && permutation->size() != ndim)
                || (uniform_shape.has_value() && uniform_shape->size() != ndim))
            {
                return false;
            }
        }

        // Validate permutation if present
        if (permutation.has_value())
        {
            const auto& perm = *permutation;
            if (perm.empty())
            {
                return false;
            }

            // Check that permutation contains exactly [0, 1, ..., N-1] without copying
            std::vector<bool> seen(perm.size(), false);
            for (const auto idx : perm)
            {
                if (idx < 0 || static_cast<std::size_t>(idx) >= perm.size()
                    || seen[static_cast<std::size_t>(idx)])
                {
                    return false;
                }
                seen[static_cast<std::size_t>(idx)] = true;
            }
        }

        // Validate uniform_shape if present
        if (uniform_shape.has_value())
        {
            // Check if any specified dimension (non-null) is non-positive
            const auto has_invalid_dim = std::ranges::any_of(
                *uniform_shape,
                [](const auto& dim) { return dim.has_value() && *dim <= 0; }
            );
            if (has_invalid_dim)
            {
                return false;
            }
        }

        return true;
    }

    std::string variable_shape_tensor_extension::metadata::to_json() const
    {
        // Check if metadata is empty
        if (!dim_names.has_value() && !permutation.has_value() && !uniform_shape.has_value())
        {
            return "{}";
        }

        // Pre-calculate approximate size to minimize allocations
        std::size_t estimated_size = json_base_size;
        
        if (dim_names.has_value())
        {
            estimated_size += json_dim_names_overhead;
            for (const auto& name : *dim_names)
            {
                estimated_size += name.size() + json_string_overhead;
            }
        }
        
        if (permutation.has_value())
        {
            estimated_size += json_permutation_overhead + permutation->size() * json_integer_avg_size;
        }
        
        if (uniform_shape.has_value())
        {
            estimated_size += json_uniform_shape_overhead;
            for (const auto& dim : *uniform_shape)
            {
                estimated_size += dim.has_value() ? json_integer_avg_size : json_null_size;
            }
        }

        std::string result;
        result.reserve(estimated_size);

        auto serialize_array = [&result](const auto& arr, auto&& formatter)
        {
            result += '[';
            bool first = true;
            for (const auto& item : arr)
            {
                if (!first)
                {
                    result += ',';
                }
                first = false;
                formatter(item);
            }
            result += ']';
        };

        result += '{';
        bool first_field = true;

        if (dim_names.has_value())
        {
            if (!first_field) result += ',';
            first_field = false;
            
            result += "\"dim_names\":";
            serialize_array(
                *dim_names,
                [&result](const auto& val)
                {
                    result += '\"';
                    result += val;
                    result += '\"';
                }
            );
        }

        if (permutation.has_value())
        {
            if (!first_field) result += ',';
            first_field = false;
            
            result += "\"permutation\":";
            serialize_array(
                *permutation,
                [&result](const auto& val)
                {
                    result += std::to_string(val);
                }
            );
        }

        if (uniform_shape.has_value())
        {
            if (!first_field) result += ',';
            first_field = false;
            
            result += "\"uniform_shape\":";
            serialize_array(
                *uniform_shape,
                [&result](const auto& val)
                {
                    if (val.has_value())
                    {
                        result += std::to_string(*val);
                    }
                    else
                    {
                        result += "null";
                    }
                }
            );
        }

        result += '}';
        return result;
    }

    variable_shape_tensor_extension::metadata
    variable_shape_tensor_extension::metadata::from_json(std::string_view json)
    {
        // Handle empty or minimal JSON
        if (json.empty() || json == "{}")
        {
            return metadata{};
        }

        try
        {
            metadata result;

            simdjson::dom::parser parser;
            simdjson::dom::element doc = parser.parse(json);

            // Parse optional fields
            if (doc["dim_names"].error() == simdjson::SUCCESS)
            {
                result.dim_names = std::vector<std::string>{};
                for (auto value : doc["dim_names"].get_array())
                {
                    result.dim_names->emplace_back(value.get_string().value());
                }
            }

            if (doc["permutation"].error() == simdjson::SUCCESS)
            {
                result.permutation = std::vector<std::int64_t>{};
                for (auto value : doc["permutation"].get_array())
                {
                    result.permutation->push_back(static_cast<std::int64_t>(value.get_int64()));
                }
            }

            if (doc["uniform_shape"].error() == simdjson::SUCCESS)
            {
                result.uniform_shape = std::vector<std::optional<std::int32_t>>{};
                for (auto value : doc["uniform_shape"].get_array())
                {
                    if (value.is_null())
                    {
                        result.uniform_shape->push_back(std::nullopt);
                    }
                    else
                    {
                        result.uniform_shape->push_back(static_cast<std::int32_t>(value.get_int64()));
                    }
                }
            }

            if (!result.is_valid())
            {
                throw std::runtime_error("Invalid metadata");
            }

            return result;
        }
        catch (const simdjson::simdjson_error& e)
        {
            throw std::runtime_error(std::string("JSON parsing error: ") + e.what());
        }
    }

    void variable_shape_tensor_extension::init(sparrow::arrow_proxy& proxy, const metadata& tensor_metadata)
    {
        SPARROW_ASSERT_TRUE(tensor_metadata.is_valid());

        // Get existing metadata
        auto existing_metadata = proxy.metadata();
        std::vector<sparrow::metadata_pair> extension_metadata;

        if (existing_metadata.has_value())
        {
            extension_metadata.assign(existing_metadata->begin(), existing_metadata->end());

            // Check if extension metadata already exists
            const bool has_extension_name = std::ranges::find_if(
                                                extension_metadata,
                                                [](const auto& pair)
                                                {
                                                    return pair.first == "ARROW:extension:name"
                                                           && pair.second == EXTENSION_NAME;
                                                }
                                            )
                                            != extension_metadata.end();

            if (has_extension_name)
            {
                proxy.set_metadata(std::make_optional(std::move(extension_metadata)));
                return;
            }
        }

        // Reserve space for new entries
        extension_metadata.reserve(extension_metadata.size() + 2);
        extension_metadata.emplace_back("ARROW:extension:name", std::string(EXTENSION_NAME));
        extension_metadata.emplace_back("ARROW:extension:metadata", tensor_metadata.to_json());

        proxy.set_metadata(std::make_optional(std::move(extension_metadata)));
    }

    variable_shape_tensor_extension::metadata
    variable_shape_tensor_extension::extract_metadata(const sparrow::arrow_proxy& proxy)
    {
        const auto metadata_opt = proxy.metadata();
        if (!metadata_opt.has_value())
        {
            return metadata{};
        }

        // Find the extension metadata entry
        const auto it = std::ranges::find_if(
            *metadata_opt,
            [](const auto& pair) { return pair.first == "ARROW:extension:metadata"; }
        );
        
        return (it != metadata_opt->end()) ? metadata::from_json((*it).second) : metadata{};
    }

    // variable_shape_tensor_array implementation

    variable_shape_tensor_array::variable_shape_tensor_array(sparrow::arrow_proxy proxy)
        : m_storage(proxy)
        , m_metadata(variable_shape_tensor_extension::extract_metadata(proxy))
    {
        SPARROW_ASSERT_TRUE(m_metadata.is_valid());
    }

    variable_shape_tensor_array::variable_shape_tensor_array(
        std::uint64_t ndim,
        sparrow::array&& tensor_data,
        sparrow::array&& tensor_shapes,
        const metadata_type& tensor_metadata
    )
        : m_storage(detail::make_tensor_struct(std::move(tensor_data), std::move(tensor_shapes)))
        , m_metadata(tensor_metadata)
    {
        validate_and_init(ndim);
    }

    variable_shape_tensor_array::variable_shape_tensor_array(
        std::uint64_t ndim,
        sparrow::array&& tensor_data,
        sparrow::array&& tensor_shapes,
        const metadata_type& tensor_metadata,
        std::string_view name,
        std::optional<std::vector<sparrow::metadata_pair>> arrow_metadata
    )
        : m_storage(detail::make_tensor_struct(std::move(tensor_data), std::move(tensor_shapes)))
        , m_metadata(tensor_metadata)
    {
        validate_and_init(ndim, name, arrow_metadata.has_value() ? &arrow_metadata : nullptr);
    }

    auto variable_shape_tensor_array::size() const -> size_type
    {
        return m_storage.size();
    }

    auto variable_shape_tensor_array::get_metadata() const -> const metadata_type&
    {
        return m_metadata;
    }

    std::optional<std::size_t> variable_shape_tensor_array::ndim() const
    {
        return m_metadata.get_ndim();
    }

    const sparrow::struct_array& variable_shape_tensor_array::storage() const
    {
        return m_storage;
    }

    sparrow::struct_array& variable_shape_tensor_array::storage()
    {
        return m_storage;
    }

    auto variable_shape_tensor_array::get_arrow_proxy() const -> const sparrow::arrow_proxy&
    {
        return sparrow::detail::array_access::get_arrow_proxy(m_storage);
    }

    auto variable_shape_tensor_array::get_arrow_proxy() -> sparrow::arrow_proxy&
    {
        return sparrow::detail::array_access::get_arrow_proxy(m_storage);
    }

    void variable_shape_tensor_array::validate_and_init(
        std::uint64_t ndim,
        std::optional<std::string_view> name,
        std::optional<std::vector<sparrow::metadata_pair>>* arrow_metadata
    )
    {
        SPARROW_ASSERT_TRUE(m_metadata.is_valid());
        
        // Validate ndim if metadata provides it
        if (const auto metadata_ndim = m_metadata.get_ndim(); metadata_ndim.has_value())
        {
            SPARROW_ASSERT_TRUE(ndim == *metadata_ndim);
        }

        auto& proxy = sparrow::detail::array_access::get_arrow_proxy(m_storage);
        
        if (name.has_value())
        {
            proxy.set_name(*name);
        }
        
        if (arrow_metadata != nullptr && arrow_metadata->has_value())
        {
            proxy.set_metadata(std::make_optional(**arrow_metadata));
        }

        variable_shape_tensor_extension::init(proxy, m_metadata);
    }

    const sparrow::array_wrapper* variable_shape_tensor_array::data_child() const
    {
        return m_storage.raw_child(0);
    }

    sparrow::array_wrapper* variable_shape_tensor_array::data_child()
    {
        return m_storage.raw_child(0);
    }

    const sparrow::array_wrapper* variable_shape_tensor_array::shape_child() const
    {
        return m_storage.raw_child(1);
    }

    sparrow::array_wrapper* variable_shape_tensor_array::shape_child()
    {
        return m_storage.raw_child(1);
    }

    auto variable_shape_tensor_array::at(size_type i) const -> const_reference
    {
        if (i >= size())
        {
            throw std::out_of_range("variable_shape_tensor_array::at: index out of range");
        }
        return m_storage[i];
    }

    bool variable_shape_tensor_array::is_valid() const
    {
        return m_storage.children_count() == 2 && m_metadata.is_valid();
    }

}  // namespace sparrow_extensions

namespace sparrow::detail
{
    SPARROW_EXTENSIONS_API const bool variable_shape_tensor_array_registered = []()
    {
        auto& registry = array_registry::instance();

        registry.register_extension(
            data_type::STRUCT,
            "arrow.variable_shape_tensor",
            [](arrow_proxy proxy)
            {
                return cloning_ptr<array_wrapper>{
                    new array_wrapper_impl<sparrow_extensions::variable_shape_tensor_array>(
                        sparrow_extensions::variable_shape_tensor_array(std::move(proxy))
                    )
                };
            }
        );

        return true;
    }();
}
