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

#include <algorithm>
#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include <sparrow/array.hpp>

#include <sparrow/primitive_array.hpp>
#include <sparrow/types/data_type.hpp>

#include "sparrow_extensions/fixed_shape_tensor.hpp"

namespace sparrow_extensions
{
    TEST_SUITE("fixed_shape_tensor")
    {
        using metadata = fixed_shape_tensor_extension::metadata;

        TEST_CASE("metadata::is_valid")
            {
                SUBCASE("valid simple shape")
                {
                    metadata meta{{2, 3}, std::nullopt, std::nullopt};
                    CHECK(meta.is_valid());
                }

                SUBCASE("valid with dim_names")
                {
                    metadata meta{{100, 200, 500}, std::vector<std::string>{"C", "H", "W"}, std::nullopt};
                    CHECK(meta.is_valid());
                }

                SUBCASE("valid with permutation")
                {
                    metadata meta{{100, 200, 500}, std::nullopt, std::vector<std::int64_t>{2, 0, 1}};
                    CHECK(meta.is_valid());
                }

                SUBCASE("valid with both dim_names and permutation")
                {
                    metadata meta{
                        {100, 200, 500},
                        std::vector<std::string>{"C", "H", "W"},
                        std::vector<std::int64_t>{2, 0, 1}
                    };
                    CHECK(meta.is_valid());
                }

                SUBCASE("invalid empty shape")
                {
                    metadata meta{{}, std::nullopt, std::nullopt};
                    CHECK_FALSE(meta.is_valid());
                }

                SUBCASE("invalid negative dimension")
                {
                    metadata meta{{2, -3}, std::nullopt, std::nullopt};
                    CHECK_FALSE(meta.is_valid());
                }

                SUBCASE("invalid zero dimension")
                {
                    metadata meta{{2, 0, 4}, std::nullopt, std::nullopt};
                    CHECK_FALSE(meta.is_valid());
                }

                SUBCASE("invalid dim_names size mismatch")
                {
                    metadata meta{{100, 200}, std::vector<std::string>{"C", "H", "W"}, std::nullopt};
                    CHECK_FALSE(meta.is_valid());
                }

                SUBCASE("invalid permutation size mismatch")
                {
                    metadata meta{{100, 200, 500}, std::nullopt, std::vector<std::int64_t>{2, 0}};
                    CHECK_FALSE(meta.is_valid());
                }

                SUBCASE("invalid permutation values")
                {
                    metadata meta{{100, 200, 500}, std::nullopt, std::vector<std::int64_t>{0, 0, 1}};
                    CHECK_FALSE(meta.is_valid());
                }

                SUBCASE("invalid permutation out of range")
                {
                    metadata meta{{100, 200, 500}, std::nullopt, std::vector<std::int64_t>{0, 1, 3}};
                    CHECK_FALSE(meta.is_valid());
                }
            }

            TEST_CASE("compute_size")
            {
                SUBCASE("simple 2D")
                {
                    metadata meta{{2, 5}, std::nullopt, std::nullopt};
                    CHECK_EQ(meta.compute_size(), 10);
                }

                SUBCASE("3D tensor")
                {
                    metadata meta{{100, 200, 500}, std::nullopt, std::nullopt};
                    CHECK_EQ(meta.compute_size(), 10000000);
                }

                SUBCASE("1D tensor")
                {
                    metadata meta{{42}, std::nullopt, std::nullopt};
                    CHECK_EQ(meta.compute_size(), 42);
                }

                SUBCASE("4D tensor")
                {
                    metadata meta{{2, 3, 4, 5}, std::nullopt, std::nullopt};
                    CHECK_EQ(meta.compute_size(), 120);
                }
            }

            TEST_CASE("to_json")
            {
                SUBCASE("simple shape")
                {
                    metadata meta{{2, 5}, std::nullopt, std::nullopt};
                    const std::string json = meta.to_json();
                    CHECK_EQ(json, R"({"shape":[2,5]})");
                }

                SUBCASE("with dim_names")
                {
                    metadata meta{{100, 200, 500}, std::vector<std::string>{"C", "H", "W"}, std::nullopt};
                    const std::string json = meta.to_json();
                    CHECK_EQ(json, R"({"shape":[100,200,500],"dim_names":["C","H","W"]})");
                }

                SUBCASE("with permutation")
                {
                    metadata meta{{100, 200, 500}, std::nullopt, std::vector<std::int64_t>{2, 0, 1}};
                    const std::string json = meta.to_json();
                    CHECK_EQ(json, R"({"shape":[100,200,500],"permutation":[2,0,1]})");
                }

                SUBCASE("with both dim_names and permutation")
                {
                    metadata meta{
                        {100, 200, 500},
                        std::vector<std::string>{"C", "H", "W"},
                        std::vector<std::int64_t>{2, 0, 1}
                    };
                    const std::string json = meta.to_json();
                    CHECK_EQ(
                        json,
                        R"({"shape":[100,200,500],"dim_names":["C","H","W"],"permutation":[2,0,1]})"
                    );
                }
            }

            TEST_CASE("from_json")
            {
                SUBCASE("simple shape")
                {
                    const std::string json = R"({"shape":[2,5]})";
                    const metadata meta = metadata::from_json(json);
                    CHECK(meta.is_valid());
                    REQUIRE_EQ(meta.shape.size(), 2);
                    CHECK_EQ(meta.shape[0], 2);
                    CHECK_EQ(meta.shape[1], 5);
                    CHECK_FALSE(meta.dim_names.has_value());
                    CHECK_FALSE(meta.permutation.has_value());
                }

                SUBCASE("with dim_names")
                {
                    const std::string json = R"({"shape":[100,200,500],"dim_names":["C","H","W"]})";
                    const metadata meta = metadata::from_json(json);
                    CHECK(meta.is_valid());
                    REQUIRE_EQ(meta.shape.size(), 3);
                    CHECK_EQ(meta.shape[0], 100);
                    CHECK_EQ(meta.shape[1], 200);
                    CHECK_EQ(meta.shape[2], 500);
                    REQUIRE(meta.dim_names.has_value());
                    REQUIRE_EQ(meta.dim_names->size(), 3);
                    CHECK_EQ((*meta.dim_names)[0], "C");
                    CHECK_EQ((*meta.dim_names)[1], "H");
                    CHECK_EQ((*meta.dim_names)[2], "W");
                    CHECK_FALSE(meta.permutation.has_value());
                }

                SUBCASE("with permutation")
                {
                    const std::string json = R"({"shape":[100,200,500],"permutation":[2,0,1]})";
                    const metadata meta = metadata::from_json(json);
                    CHECK(meta.is_valid());
                    REQUIRE_EQ(meta.shape.size(), 3);
                    CHECK_FALSE(meta.dim_names.has_value());
                    REQUIRE(meta.permutation.has_value());
                    REQUIRE_EQ(meta.permutation->size(), 3);
                    CHECK_EQ((*meta.permutation)[0], 2);
                    CHECK_EQ((*meta.permutation)[1], 0);
                    CHECK_EQ((*meta.permutation)[2], 1);
                }

                SUBCASE("with whitespace")
                {
                    const std::string json = R"(  {  "shape"  : [ 2 , 5 ]  }  )";
                    const metadata meta = metadata::from_json(json);
                    CHECK(meta.is_valid());
                    REQUIRE_EQ(meta.shape.size(), 2);
                    CHECK_EQ(meta.shape[0], 2);
                    CHECK_EQ(meta.shape[1], 5);
                }

                SUBCASE("invalid - missing shape")
                {
                    const std::string json = R"({"dim_names":["C","H","W"]})";
                    CHECK_THROWS_AS(metadata::from_json(json), std::runtime_error);
                }

                SUBCASE("invalid - malformed JSON")
                {
                    const std::string json = R"({"shape":[2,5)";
                    CHECK_THROWS_AS(metadata::from_json(json), std::runtime_error);
                }
            }

            TEST_CASE("round-trip serialization")
            {
                SUBCASE("simple")
                {
                    metadata original{{2, 5}, std::nullopt, std::nullopt};
                    const std::string json = original.to_json();
                    const metadata parsed = metadata::from_json(json);
                    CHECK(parsed.shape == original.shape);
                    CHECK(parsed.dim_names == original.dim_names);
                    CHECK(parsed.permutation == original.permutation);
                }

                SUBCASE("complex")
                {
                    metadata original{
                        {100, 200, 500},
                        std::vector<std::string>{"C", "H", "W"},
                        std::vector<std::int64_t>{2, 0, 1}
                    };
                    const std::string json = original.to_json();
                    const metadata parsed = metadata::from_json(json);
                    CHECK(parsed.shape == original.shape);
                    CHECK(parsed.dim_names == original.dim_names);
                    CHECK(parsed.permutation == original.permutation);
                }
            }

        TEST_CASE("fixed_shape_tensor_array::constructor with simple 2D tensors")
            {
                // Create a flattened array of 3 tensors of shape [2, 3]
                // Total elements: 3 * 2 * 3 = 18
                std::vector<float> flat_data;
                for (int i = 0; i < 18; ++i)
                {
                    flat_data.push_back(static_cast<float>(i));
                }

                sparrow::primitive_array<float> values_array(flat_data);
                const std::vector<std::int64_t> shape{2, 3};
                metadata tensor_meta{shape, std::nullopt, std::nullopt};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array tensor_array(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                CHECK_EQ(tensor_array.size(), 3);
                CHECK(tensor_array.shape() == shape);

                const auto& retrieved_meta = tensor_array.get_metadata();
                CHECK(retrieved_meta.shape == shape);
                CHECK_FALSE(retrieved_meta.dim_names.has_value());
                CHECK_FALSE(retrieved_meta.permutation.has_value());
            }

            TEST_CASE("constructor with 3D tensors and dim_names")
            {
                // Create 2 tensors of shape [2, 2, 2]
                // Total elements: 2 * 2 * 2 * 2 = 16
                std::vector<int32_t> flat_data(16);
                std::iota(flat_data.begin(), flat_data.end(), 0);

                sparrow::primitive_array<int32_t> values_array(flat_data);
                const std::vector<std::int64_t> shape{2, 2, 2};
                const std::vector<std::string> dim_names{"X", "Y", "Z"};
                metadata tensor_meta{shape, dim_names, std::nullopt};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array tensor_array(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                CHECK_EQ(tensor_array.size(), 2);
                CHECK(tensor_array.shape() == shape);

                const auto& meta = tensor_array.get_metadata();
                CHECK(meta.shape == shape);
                REQUIRE(meta.dim_names.has_value());
                CHECK(*meta.dim_names == dim_names);
                CHECK_FALSE(meta.permutation.has_value());
            }

            TEST_CASE("constructor with permutation")
            {
                // Create 1 tensor of shape [3, 4, 5]
                // Total elements: 60
                std::vector<double> flat_data(60);
                std::iota(flat_data.begin(), flat_data.end(), 0.0);

                sparrow::primitive_array<double> values_array(flat_data);
                const std::vector<std::int64_t> shape{3, 4, 5};
                const std::vector<std::int64_t> permutation{2, 0, 1};  // Logical shape is [5, 3, 4]
                metadata tensor_meta{shape, std::nullopt, permutation};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array tensor_array(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                CHECK_EQ(tensor_array.size(), 1);
                CHECK(tensor_array.shape() == shape);

                const auto& meta = tensor_array.get_metadata();
                CHECK(meta.shape == shape);
                CHECK_FALSE(meta.dim_names.has_value());
                REQUIRE(meta.permutation.has_value());
                CHECK(*meta.permutation == permutation);
            }

            TEST_CASE("constructor with validity bitmap")
            {
                // Create 4 tensors of shape [2, 2]
                // Total elements: 4 * 2 * 2 = 16
                std::vector<int32_t> flat_data(16);
                std::iota(flat_data.begin(), flat_data.end(), 0);

                sparrow::primitive_array<int32_t> values_array(flat_data);
                const std::vector<std::int64_t> shape{2, 2};
                metadata tensor_meta{shape, std::nullopt, std::nullopt};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array tensor_array(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                CHECK_EQ(tensor_array.size(), 4);

                // When nullable is true but no validity bitmap is provided,
                // all elements are valid by default
                const auto& storage = tensor_array.storage();
                CHECK(storage[0].has_value());
                CHECK(storage[1].has_value());
                CHECK(storage[2].has_value());
                CHECK(storage[3].has_value());
            }

            TEST_CASE("element access")
            {
                // Create 2 tensors of shape [2, 3]
                std::vector<float> flat_data{
                    // First tensor
                    1.0f,
                    2.0f,
                    3.0f,
                    4.0f,
                    5.0f,
                    6.0f,
                    // Second tensor
                    7.0f,
                    8.0f,
                    9.0f,
                    10.0f,
                    11.0f,
                    12.0f
                };

                sparrow::primitive_array<float> values_array(flat_data);
                const std::vector<std::int64_t> shape{2, 3};
                metadata tensor_meta{shape, std::nullopt, std::nullopt};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array tensor_array(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                // Access first tensor
                auto tensor0 = tensor_array[0];
                CHECK(tensor0.has_value());

                // Access second tensor
                auto tensor1 = tensor_array[1];
                CHECK(tensor1.has_value());
            }

            TEST_CASE("1D tensor (vector)")
            {
                // Create 5 vectors of length 10
                std::vector<int32_t> flat_data(50);
                std::iota(flat_data.begin(), flat_data.end(), 0);

                sparrow::primitive_array<int32_t> values_array(flat_data);
                const std::vector<std::int64_t> shape{10};
                metadata tensor_meta{shape, std::nullopt, std::nullopt};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array tensor_array(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                CHECK_EQ(tensor_array.size(), 5);
                CHECK_EQ(tensor_array.shape()[0], 10);
            }

            TEST_CASE("extension metadata roundtrip")
            {
                // Create array with metadata
                std::vector<float> flat_data(12);
                std::iota(flat_data.begin(), flat_data.end(), 0.0f);

                sparrow::primitive_array<float> values_array(flat_data);
                const std::vector<std::int64_t> shape{2, 3};
                const std::vector<std::string> dim_names{"rows", "cols"};
                const std::vector<std::int64_t> permutation{1, 0};
                metadata tensor_meta{shape, dim_names, permutation};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array tensor_array(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                // Get and verify the metadata
                const auto& extracted_meta = tensor_array.get_metadata();

                // Verify all fields
                CHECK(extracted_meta.shape == shape);
                REQUIRE(extracted_meta.dim_names.has_value());
                CHECK(*extracted_meta.dim_names == dim_names);
                REQUIRE(extracted_meta.permutation.has_value());
                CHECK(*extracted_meta.permutation == permutation);
            }

            TEST_CASE("copy constructor")
            {
                std::vector<int32_t> flat_data{1, 2, 3, 4, 5, 6};
                sparrow::primitive_array<int32_t> values_array(flat_data);
                const std::vector<std::int64_t> shape{2, 3};
                metadata tensor_meta{shape, std::nullopt, std::nullopt};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array original(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                fixed_shape_tensor_array copy(original);

                CHECK_EQ(copy.size(), original.size());
                CHECK(copy.shape() == original.shape());
                CHECK(copy.get_metadata().shape == original.get_metadata().shape);
            }

            TEST_CASE("move constructor")
            {
                std::vector<int32_t> flat_data{1, 2, 3, 4, 5, 6};
                sparrow::primitive_array<int32_t> values_array(flat_data);
                const std::vector<std::int64_t> shape{2, 3};
                metadata tensor_meta{shape, std::nullopt, std::nullopt};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array original(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                const std::size_t original_size = original.size();
                const auto original_shape = original.shape();

                fixed_shape_tensor_array moved(std::move(original));

                CHECK_EQ(moved.size(), original_size);
                CHECK(moved.shape() == original_shape);
            }

            TEST_CASE("storage access")
            {
                std::vector<float> flat_data(6);
                std::iota(flat_data.begin(), flat_data.end(), 0.0f);

                sparrow::primitive_array<float> values_array(flat_data);
                const std::vector<std::int64_t> shape{2, 3};
                metadata tensor_meta{shape, std::nullopt, std::nullopt};
                const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                fixed_shape_tensor_array tensor_array(
                    list_size,
                    sparrow::array(std::move(values_array)),
                    tensor_meta,
                    true
                );

                // Test const access
                const auto& const_storage = tensor_array.storage();
                CHECK_EQ(const_storage.size(), 1);

                // Test mutable access
                auto& mut_storage = tensor_array.storage();
                CHECK_EQ(mut_storage.size(), 1);
            }

            TEST_CASE("spec examples")
            {
                SUBCASE("Example: { \"shape\": [2, 5]}")
                {
                    std::vector<double> flat_data(10);  // 1 tensor of shape [2, 5]
                    std::iota(flat_data.begin(), flat_data.end(), 0.0);

                    sparrow::primitive_array<double> values_array(flat_data);
                    const std::vector<std::int64_t> shape{2, 5};
                    metadata tensor_meta{shape, std::nullopt, std::nullopt};
                    const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                    fixed_shape_tensor_array tensor_array(
                        list_size,
                        sparrow::array(std::move(values_array)),
                        tensor_meta,
                        true
                    );

                    CHECK_EQ(tensor_array.size(), 1);
                    CHECK(tensor_array.shape() == shape);
                    CHECK_EQ(tensor_array.get_metadata().compute_size(), 10);
                }

                SUBCASE("Example with dim_names for NCHW: { \"shape\": [100, 200, 500], \"dim_names\": "
                        "[\"C\", \"H\", \"W\"]}")
                {
                    // Just one tensor for testing
                    const std::int64_t tensor_size = 100 * 200 * 500;
                    std::vector<float> flat_data(tensor_size, 0.0f);

                    sparrow::primitive_array<float> values_array(flat_data);
                    const std::vector<std::int64_t> shape{100, 200, 500};
                    const std::vector<std::string> dim_names{"C", "H", "W"};
                    metadata tensor_meta{shape, dim_names, std::nullopt};
                    const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                    fixed_shape_tensor_array tensor_array(
                        list_size,
                        sparrow::array(std::move(values_array)),
                        tensor_meta,
                        true
                    );

                    CHECK_EQ(tensor_array.size(), 1);
                    CHECK(tensor_array.shape() == shape);
                    const auto& retrieved_meta = tensor_array.get_metadata();
                    REQUIRE(retrieved_meta.dim_names.has_value());
                    CHECK(*retrieved_meta.dim_names == dim_names);
                }

                SUBCASE("Example with permutation: { \"shape\": [100, 200, 500], \"permutation\": [2, 0, "
                        "1]}")
                {
                    // Physical shape [100, 200, 500], logical shape [500, 100, 200]
                    const std::int64_t tensor_size = 100 * 200 * 500;
                    std::vector<float> flat_data(tensor_size, 0.0f);

                    sparrow::primitive_array<float> values_array(flat_data);
                    const std::vector<std::int64_t> shape{100, 200, 500};
                    const std::vector<std::int64_t> permutation{2, 0, 1};
                    metadata tensor_meta{shape, std::nullopt, permutation};
                    const std::uint64_t list_size = static_cast<std::uint64_t>(tensor_meta.compute_size());

                    fixed_shape_tensor_array tensor_array(
                        list_size,
                        sparrow::array(std::move(values_array)),
                        tensor_meta,
                        true
                    );

                    CHECK_EQ(tensor_array.size(), 1);
                    CHECK(tensor_array.shape() == shape);  // Physical shape
                    const auto& retrieved_meta = tensor_array.get_metadata();
                    REQUIRE(retrieved_meta.permutation.has_value());
                    CHECK(*retrieved_meta.permutation == permutation);

                    // Note: Logical shape would be [500, 100, 200]
                    // which is shape[permutation[i]] for each i
                }
            }
    }
}  // namespace sparrow_extensions
