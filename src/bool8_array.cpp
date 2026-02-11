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

#include "sparrow_extensions/bool8_array.hpp"

#include "sparrow/layout/array_registry.hpp"

#include "sparrow_extensions/config/config.hpp"

namespace sparrow_extensions::detail
{
    SPARROW_EXTENSIONS_API const bool bool8_array_registered = []()
    {
        auto& registry = sparrow::array_registry::instance();

        registry.register_extension(
            sparrow::data_type::INT8,
            "arrow.bool8",
            [](sparrow::arrow_proxy proxy)
            {
                return sparrow::cloning_ptr<sparrow::array_wrapper>{
                    new sparrow::array_wrapper_impl<bool8_array>(bool8_array(std::move(proxy)))
                };
            }
        );

        return true;
    }();
}
