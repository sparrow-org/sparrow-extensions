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

#include "sparrow_extensions/uuid_array.hpp"

#include "sparrow/layout/array_registry.hpp"

#include "sparrow_extensions/config/config.hpp"

namespace sparrow::detail
{
    SPARROW_EXTENSIONS_API const bool uuid_array_registered = []()
    {
        auto& registry = array_registry::instance();

        registry.register_extension(
            data_type::FIXED_WIDTH_BINARY,
            "arrow.uuid",
            [](arrow_proxy proxy)
            {
                return cloning_ptr<array_wrapper>{new array_wrapper_impl<sparrow_extensions::uuid_array>(
                    sparrow_extensions::uuid_array(std::move(proxy))
                )};
            }
        );

        return true;
    }();
}
