//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_test_forall_rangestride_HPP
#define RAJA_test_forall_rangestride_HPP

#include <cstdlib>
#include <string>

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

constexpr std::size_t LENGTH{1024};

template <typename TestPolicy>
class ForallRangeStrideFunctionalTest : public ::testing::Test
{
  using Allocator = RAJA::util::allocator< 
    //RAJA::Platform::host
    RAJA::detail::get_platform<TestPolicy>::value >;

protected:
  virtual void SetUp()
  {
    array = Allocator::allocate<int>(LENGTH);
    for (RAJA::Index_type i = 0; i < LENGTH; ++i) {
      array[i] = rand() % 65536;
    }
  }

  virtual void TearDown()
  {
    Allocator::deallocate(array);
  }

  int* array;
};

TYPED_TEST_CASE_P(ForallRangeStrideFunctionalTest);

TYPED_TEST_P(ForallRangeStrideFunctionalTest, ForallUnitStride)
{
  RAJA::RangeStrideSegment seg{0, LENGTH, 1};

  RAJA::forall<TypeParam>(seg, [=](RAJA::Index_type idx) {
    this->array[idx] = idx;
  });

  int* result = Allocator::get(this->array);

  for (RAJA::Index_type i = 0; i < LENGTH; ++i) {
    EXPECT_EQ(result[i], i);
  }
}

TYPED_TEST_P(ForallRangeStrideFunctionalTest, ForallUnitNegativeStride)
{
  RAJA::RangeStrideSegment seg{LENGTH-1, -1, -11};

  RAJA::forall<TypeParam>(seg, [=](RAJA::Index_type idx) {
    this->array[idx] = idx;
  });

  int* result = Allocator::get(this->array);

  for (RAJA::Index_type i = 0; i < LENGTH; ++i) {
    EXPECT_EQ(result[i], i);
  }
}

REGISTER_TYPED_TEST_CASE_P(
    ForallRangeStrideFunctionalTest, 
    ForallUnitStride,
    ForallUnitNegativeStride);

#endif // RAJA_test-forall-rangestride_HPP
