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
#include <sstream>
#include <stdexcept>

#include "sparrow/layout/array_access.hpp"
#include "sparrow/layout/array_registry.hpp"
#include "sparrow/utils/contracts.hpp"

#include "sparrow_extensions/config/config.hpp"

namespace sparrow_extensions
{
    // Metadata implementation
    bool fixed_shape_tensor_extension::metadata::is_valid() const
    {
        // Shape must not be empty and all dimensions must be positive
        if (shape.empty())
        {
            return false;
        }

        for (const auto dim : shape)
        {
            if (dim <= 0)
            {
                return false;
            }
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

            // Check that permutation contains exactly [0, 1, ..., N-1]
            std::vector<std::int64_t> sorted_perm = perm;
            std::ranges::sort(sorted_perm);
            for (std::size_t i = 0; i < sorted_perm.size(); ++i)
            {
                if (sorted_perm[i] != static_cast<std::int64_t>(i))
                {
                    return false;
                }
            }
        }

        return true;
    }

    std::int64_t fixed_shape_tensor_extension::metadata::compute_size() const
    {
        return std::accumulate(
            shape.begin(),
            shape.end(),
            std::int64_t{1},
            std::multiplies<std::int64_t>{}
        );
    }

    std::string fixed_shape_tensor_extension::metadata::to_json() const
    {
        std::ostringstream oss;
        oss << "{\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (i > 0)
            {
                oss << ",";
            }
            oss << shape[i];
        }
        oss << "]";

        if (dim_names.has_value())
        {
            oss << ",\"dim_names\":[";
            for (std::size_t i = 0; i < dim_names->size(); ++i)
            {
                if (i > 0)
                {
                    oss << ",";
                }
                oss << "\"" << (*dim_names)[i] << "\"";
            }
            oss << "]";
        }

        if (permutation.has_value())
        {
            oss << ",\"permutation\":[";
            for (std::size_t i = 0; i < permutation->size(); ++i)
            {
                if (i > 0)
                {
                    oss << ",";
                }
                oss << (*permutation)[i];
            }
            oss << "]";
        }

        oss << "}";
        return oss.str();
    }

    fixed_shape_tensor_extension::metadata fixed_shape_tensor_extension::metadata::from_json(
        std::string_view json
    )
    {
        metadata result;

        // Simple JSON parser for the fixed structure we expect
        // This is a minimal implementation - production code might use a proper JSON library

        std::string json_str(json);
        std::size_t pos = 0;

        // Helper to skip whitespace
        auto skip_whitespace = [&]()
        {
            while (pos < json_str.size() && std::isspace(json_str[pos]))
            {
                ++pos;
            }
        };

        // Helper to read a string value
        auto read_string = [&]() -> std::string
        {
            skip_whitespace();
            if (pos >= json_str.size() || json_str[pos] != '"')
            {
                throw std::runtime_error("Expected opening quote");
            }
            ++pos;  // Skip opening quote

            std::size_t start = pos;
            while (pos < json_str.size() && json_str[pos] != '"')
            {
                ++pos;
            }

            if (pos >= json_str.size())
            {
                throw std::runtime_error("Expected closing quote");
            }

            std::string value = json_str.substr(start, pos - start);
            ++pos;  // Skip closing quote
            return value;
        };

        // Helper to read an integer
        auto read_int = [&]() -> std::int64_t
        {
            skip_whitespace();
            std::size_t start = pos;
            if (pos < json_str.size() && (json_str[pos] == '-' || json_str[pos] == '+'))
            {
                ++pos;
            }
            while (pos < json_str.size() && std::isdigit(json_str[pos]))
            {
                ++pos;
            }
            return std::stoll(json_str.substr(start, pos - start));
        };

        // Helper to read an array of integers
        auto read_int_array = [&]() -> std::vector<std::int64_t>
        {
            std::vector<std::int64_t> arr;
            skip_whitespace();
            if (pos >= json_str.size() || json_str[pos] != '[')
            {
                throw std::runtime_error("Expected opening bracket");
            }
            ++pos;

            skip_whitespace();
            if (pos < json_str.size() && json_str[pos] == ']')
            {
                ++pos;
                return arr;
            }

            while (true)
            {
                arr.push_back(read_int());
                skip_whitespace();

                if (pos >= json_str.size())
                {
                    throw std::runtime_error("Unexpected end of JSON");
                }

                if (json_str[pos] == ']')
                {
                    ++pos;
                    break;
                }

                if (json_str[pos] != ',')
                {
                    throw std::runtime_error("Expected comma or closing bracket");
                }
                ++pos;
            }

            return arr;
        };

        // Helper to read an array of strings
        auto read_string_array = [&]() -> std::vector<std::string>
        {
            std::vector<std::string> arr;
            skip_whitespace();
            if (pos >= json_str.size() || json_str[pos] != '[')
            {
                throw std::runtime_error("Expected opening bracket");
            }
            ++pos;

            skip_whitespace();
            if (pos < json_str.size() && json_str[pos] == ']')
            {
                ++pos;
                return arr;
            }

            while (true)
            {
                arr.push_back(read_string());
                skip_whitespace();

                if (pos >= json_str.size())
                {
                    throw std::runtime_error("Unexpected end of JSON");
                }

                if (json_str[pos] == ']')
                {
                    ++pos;
                    break;
                }

                if (json_str[pos] != ',')
                {
                    throw std::runtime_error("Expected comma or closing bracket");
                }
                ++pos;
            }

            return arr;
        };

        // Parse the JSON object
        skip_whitespace();
        if (pos >= json_str.size() || json_str[pos] != '{')
        {
            throw std::runtime_error("Expected opening brace");
        }
        ++pos;

        while (true)
        {
            skip_whitespace();
            if (pos >= json_str.size())
            {
                throw std::runtime_error("Unexpected end of JSON");
            }

            if (json_str[pos] == '}')
            {
                ++pos;
                break;
            }

            std::string key = read_string();
            skip_whitespace();

            if (pos >= json_str.size() || json_str[pos] != ':')
            {
                throw std::runtime_error("Expected colon after key");
            }
            ++pos;

            if (key == "shape")
            {
                result.shape = read_int_array();
            }
            else if (key == "dim_names")
            {
                result.dim_names = read_string_array();
            }
            else if (key == "permutation")
            {
                result.permutation = read_int_array();
            }
            else
            {
                throw std::runtime_error("Unknown key: " + key);
            }

            skip_whitespace();
            if (pos >= json_str.size())
            {
                throw std::runtime_error("Unexpected end of JSON");
            }

            if (json_str[pos] == '}')
            {
                ++pos;
                break;
            }

            if (json_str[pos] != ',')
            {
                throw std::runtime_error("Expected comma or closing brace");
            }
            ++pos;
        }

        if (result.shape.empty())
        {
            throw std::runtime_error("Missing required 'shape' field");
        }

        if (!result.is_valid())
        {
            throw std::runtime_error("Invalid metadata");
        }

        return result;
    }

    void fixed_shape_tensor_extension::init(
        sparrow::arrow_proxy& proxy,
        const metadata& tensor_metadata
    )
    {
        SPARROW_ASSERT_TRUE(tensor_metadata.is_valid());

        // Get existing metadata
        std::optional<sparrow::key_value_view> existing_metadata = proxy.metadata();
        std::vector<sparrow::metadata_pair> extension_metadata =
            existing_metadata.has_value()
                ? std::vector<sparrow::metadata_pair>(
                      existing_metadata->begin(),
                      existing_metadata->end()
                  )
                : std::vector<sparrow::metadata_pair>{};

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

        if (!has_extension_name)
        {
            extension_metadata.emplace_back("ARROW:extension:name", std::string(EXTENSION_NAME));
            extension_metadata.emplace_back(
                "ARROW:extension:metadata",
                tensor_metadata.to_json()
            );
        }

        proxy.set_metadata(std::make_optional(extension_metadata));
    }

    fixed_shape_tensor_extension::metadata fixed_shape_tensor_extension::extract_metadata(
        const sparrow::arrow_proxy& proxy
    )
    {
        std::optional<sparrow::key_value_view> metadata_opt = proxy.metadata();
        if (!metadata_opt.has_value())
        {
            throw std::runtime_error("Missing extension metadata");
        }

        const auto& metadata = *metadata_opt;
        std::string metadata_json;

        for (const auto& [key, value] : metadata)
        {
            if (key == "ARROW:extension:metadata")
            {
                metadata_json = value;
                break;
            }
        }

        if (metadata_json.empty())
        {
            throw std::runtime_error("Missing ARROW:extension:metadata");
        }

        return metadata::from_json(metadata_json);
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
        const metadata_type& tensor_metadata,
        bool nullable
    )
        : m_storage(list_size, std::move(flat_values), nullable)
        , m_metadata(tensor_metadata)
    {
        SPARROW_ASSERT_TRUE(m_metadata.is_valid());
        SPARROW_ASSERT_TRUE(static_cast<std::int64_t>(list_size) == m_metadata.compute_size());
        SPARROW_ASSERT_TRUE(m_storage.size() * list_size == (flat_values.size() ? flat_values.size() : m_storage.size() * list_size)); 

        // Add extension metadata to the storage using array_access
        fixed_shape_tensor_extension::init(
            sparrow::detail::array_access::get_arrow_proxy(m_storage),
            m_metadata
        );
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

    auto fixed_shape_tensor_array::storage() const -> const sparrow::fixed_sized_list_array&
    {
        return m_storage;
    }

    auto fixed_shape_tensor_array::storage() -> sparrow::fixed_sized_list_array&
    {
        return m_storage;
    }

    auto fixed_shape_tensor_array::operator[](size_type i) const -> decltype(std::declval<const sparrow::fixed_sized_list_array&>()[i])
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
