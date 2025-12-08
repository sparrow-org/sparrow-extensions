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

#include "sparrow_extensions/json_array.hpp"

#include <sparrow/layout/array_registry.hpp>

#include "sparrow_extensions/config/config.hpp"

namespace sparrow_extensions::detail
{
    SPARROW_EXTENSIONS_API const bool json_arrays_registered = []()
    {
        auto& registry = sparrow::array_registry::instance();

        constexpr std::string_view extension_name = "arrow.json";

        // Register json_array (STRING base type)
        registry.register_extension(
            sparrow::data_type::STRING,
            extension_name,
            [](sparrow::arrow_proxy proxy)
            {
                return sparrow::cloning_ptr<sparrow::array_wrapper>{
                    new sparrow::array_wrapper_impl<json_array>(json_array(std::move(proxy)))
                };
            }
        );

        // Register big_json_array (LARGE_STRING base type)
        registry.register_extension(
            sparrow::data_type::LARGE_STRING,
            extension_name,
            [](sparrow::arrow_proxy proxy)
            {
                return sparrow::cloning_ptr<sparrow::array_wrapper>{
                    new sparrow::array_wrapper_impl<big_json_array>(big_json_array(std::move(proxy)))
                };
            }
        );

        // Register json_view_array (STRING_VIEW base type)
        registry.register_extension(
            sparrow::data_type::STRING_VIEW,
            extension_name,
            [](sparrow::arrow_proxy proxy)
            {
                return sparrow::cloning_ptr<sparrow::array_wrapper>{
                    new sparrow::array_wrapper_impl<json_view_array>(json_view_array(std::move(proxy)))
                };
            }
        );

        return true;
    }();
}
