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

#include "sparrow_extensions/fixed_shape_tensor.hpp"

#include <algorithm>
#include <numeric>
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
        constexpr std::size_t json_base_size = 10;             // {"shape":[]}
        constexpr std::size_t json_integer_avg_size = 10;      // Average size per integer
        constexpr std::size_t json_dim_names_overhead = 15;    // ,"dim_names":[]
        constexpr std::size_t json_string_overhead = 3;        // "name",
        constexpr std::size_t json_permutation_overhead = 17;  // ,"permutation":[]

        // JSON parsing capacity hints
        constexpr std::size_t typical_tensor_dimensions = 8;  // Typical tensor rank (2-4 dims, reserve 8)
    }

    // Metadata implementation
    bool fixed_shape_tensor_extension::metadata::is_valid() const
    {
        // Shape must not be empty and all dimensions must be positive
        if (shape.empty()
            || !std::ranges::all_of(
                shape,
                [](auto dim)
                {
                    return dim > 0;
                }
            ))
        {
            return false;
        }

        // If dim_names is present, it must match the shape size
        if (dim_names.has_value() && dim_names->size() != shape.size())
        {
            return false;
        }

        // If permutation is present, validate it
        if (permutation.has_value())
        {
            const auto& perm = *permutation;
            if (perm.size() != shape.size())
            {
                return false;
            }

            // Check that permutation contains exactly [0, 1, ..., N-1] without copying
            // Use a bitset to track seen indices
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
            return true;
        }

        return true;
    }

    std::int64_t fixed_shape_tensor_extension::metadata::compute_size() const
    {
        return std::reduce(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>{});
    }

    std::string fixed_shape_tensor_extension::metadata::to_json() const
    {
        // Pre-calculate approximate size to minimize allocations
        std::size_t estimated_size = json_base_size;
        estimated_size += shape.size() * json_integer_avg_size;
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

        result += "{\"shape\":";
        serialize_array(
            shape,
            [&result](const auto& val)
            {
                result += std::to_string(val);
            }
        );

        if (dim_names.has_value())
        {
            result += ",\"dim_names\":";
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
            result += ",\"permutation\":";
            serialize_array(
                *permutation,
                [&result](const auto& val)
                {
                    result += std::to_string(val);
                }
            );
        }

        result += '}';
        return result;
    }

    fixed_shape_tensor_extension::metadata
    fixed_shape_tensor_extension::metadata::from_json(std::string_view json)
    {
        auto parse_int_array = [](simdjson::ondemand::array arr) -> std::vector<std::int64_t>
        {
            std::vector<std::int64_t> result;
            result.reserve(typical_tensor_dimensions);
            for (auto value : arr)
            {
                result.push_back(static_cast<std::int64_t>(value.get_int64()));
            }
            return result;
        };

        auto parse_string_array = [](simdjson::ondemand::array arr) -> std::vector<std::string>
        {
            std::vector<std::string> result;
            result.reserve(typical_tensor_dimensions);
            for (auto value : arr)
            {
                result.emplace_back(value.get_string().value());
            }
            return result;
        };

        try
        {
            metadata result;

            simdjson::ondemand::parser parser;
            simdjson::padded_string padded_json(json);
            simdjson::ondemand::document doc = parser.iterate(padded_json);

            // Parse shape (required)
            auto shape_field = doc["shape"];
            if (shape_field.error() != simdjson::SUCCESS)
            {
                throw std::runtime_error("Missing required 'shape' field");
            }

            result.shape = parse_int_array(shape_field.get_array());

            if (result.shape.empty())
            {
                throw std::runtime_error("'shape' field cannot be empty");
            }

            // Parse optional fields
            auto dim_names_field = doc["dim_names"];
            if (dim_names_field.error() == simdjson::SUCCESS)
            {
                result.dim_names = parse_string_array(dim_names_field.get_array());
            }

            auto permutation_field = doc["permutation"];
            if (permutation_field.error() == simdjson::SUCCESS)
            {
                result.permutation = parse_int_array(permutation_field.get_array());
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

    void fixed_shape_tensor_extension::init(sparrow::arrow_proxy& proxy, const metadata& tensor_metadata)
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

    fixed_shape_tensor_extension::metadata
    fixed_shape_tensor_extension::extract_metadata(const sparrow::arrow_proxy& proxy)
    {
        const auto metadata_opt = proxy.metadata();
        if (!metadata_opt.has_value())
        {
            throw std::runtime_error("Missing extension metadata");
        }

        for (const auto& [key, value] : *metadata_opt)
        {
            if (key == "ARROW:extension:metadata")
            {
                return metadata::from_json(value);
            }
        }

        throw std::runtime_error("Missing ARROW:extension:metadata");
    }

    // fixed_shape_tensor_array implementation

    fixed_shape_tensor_array::fixed_shape_tensor_array(sparrow::arrow_proxy proxy)
        : m_storage(proxy)
        , m_metadata(fixed_shape_tensor_extension::extract_metadata(proxy))
    {
        SPARROW_ASSERT_TRUE(m_metadata.is_valid());
    }

    fixed_shape_tensor_array::fixed_shape_tensor_array(
        std::uint64_t list_size,
        sparrow::array&& flat_values,
        const metadata_type& tensor_metadata
    )
        : m_storage(list_size, std::move(flat_values), std::vector<bool>{})
        , m_metadata(tensor_metadata)
    {
        SPARROW_ASSERT_TRUE(m_metadata.is_valid());
        SPARROW_ASSERT_TRUE(static_cast<std::int64_t>(list_size) == m_metadata.compute_size());

        fixed_shape_tensor_extension::init(sparrow::detail::array_access::get_arrow_proxy(m_storage), m_metadata);
    }

    fixed_shape_tensor_array::fixed_shape_tensor_array(
        std::uint64_t list_size,
        sparrow::array&& flat_values,
        const metadata_type& tensor_metadata,
        std::string_view name,
        std::optional<std::vector<sparrow::metadata_pair>> arrow_metadata
    )
        : m_storage(list_size, std::move(flat_values), std::vector<bool>{})
        , m_metadata(tensor_metadata)
    {
        SPARROW_ASSERT_TRUE(m_metadata.is_valid());
        SPARROW_ASSERT_TRUE(static_cast<std::int64_t>(list_size) == m_metadata.compute_size());

        auto& proxy = sparrow::detail::array_access::get_arrow_proxy(m_storage);
        proxy.set_name(name);

        if (arrow_metadata.has_value())
        {
            proxy.set_metadata(std::make_optional(*arrow_metadata));
        }

        fixed_shape_tensor_extension::init(proxy, m_metadata);
    }

    auto fixed_shape_tensor_array::size() const -> size_type
    {
        return m_storage.size();
    }

    auto fixed_shape_tensor_array::get_metadata() const -> const metadata_type&
    {
        return m_metadata;
    }

    auto fixed_shape_tensor_array::shape() const -> const std::vector<std::int64_t>&
    {
        return m_metadata.shape;
    }

    const sparrow::fixed_sized_list_array& fixed_shape_tensor_array::storage() const
    {
        return m_storage;
    }

    sparrow::fixed_sized_list_array& fixed_shape_tensor_array::storage()
    {
        return m_storage;
    }

    auto fixed_shape_tensor_array::operator[](size_type i) const
        -> decltype(std::declval<const sparrow::fixed_sized_list_array&>()[i])
    {
        return m_storage[i];
    }

    auto fixed_shape_tensor_array::get_arrow_proxy() const -> const sparrow::arrow_proxy&
    {
        return sparrow::detail::array_access::get_arrow_proxy(m_storage);
    }

    auto fixed_shape_tensor_array::get_arrow_proxy() -> sparrow::arrow_proxy&
    {
        return sparrow::detail::array_access::get_arrow_proxy(m_storage);
    }

}  // namespace sparrow_extensions

namespace sparrow::detail
{
    SPARROW_EXTENSIONS_API const bool fixed_shape_tensor_array_registered = []()
    {
        auto& registry = array_registry::instance();

        registry.register_extension(
            data_type::FIXED_SIZED_LIST,
            "arrow.fixed_shape_tensor",
            [](arrow_proxy proxy)
            {
                return cloning_ptr<array_wrapper>{
                    new array_wrapper_impl<sparrow_extensions::fixed_shape_tensor_array>(
                        sparrow_extensions::fixed_shape_tensor_array(std::move(proxy))
                    )
                };
            }
        );

        return true;
    }();
}
