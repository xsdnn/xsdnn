//
// Created by shuffle on 17.08.22.
//

//#define STR(x) #x
//#define CONCAT(x,y) STR(x ## _ ## y)
//#define TEST(x,y) TEST_CASE(CONCAT(x,y))
//
//#define ASSERT_TRUE(x) REQUIRE(x)
//#define ASSERT_EQ(x,y) REQUIRE( (x) == (y) )
//#define ASSERT_NEAR(x,y,tolerance) REQUIRE( (x) == Approx(y).margin(tolerance) )
//#define ASSERT_DOUBLE_EQ(x,y) REQUIRE( (x) == Approx(y) )
//#define ASSERT_FLOAT_EQ(x,y) REQUIRE( (x) == Approx(y) )
//
//#define EXPECT_EQ(x,y) CHECK( (x) == (y) )
//#define EXPECT_STREQ(x,y) CHECK( strcmp((x),(y)) == 0 )
//#define EXPECT_NE(x,y) CHECK( (x) != (y) )
//#define EXPECT_DOUBLE_EQ(x,y) CHECK( (x) == Approx(y) )
//#define EXPECT_FLOAT_EQ(x,y) CHECK( (x) == Approx(y) )
//#define EXPECT_TRUE(x) CHECK(x)
//#define EXPECT_FALSE(x) CHECK(!(x))
//#define EXPECT_NEAR(x,y,tolerance) CHECK( (x) == Approx(y).margin(tolerance) )
//#define EXPECT_LT(x,y) CHECK((x) < (y))
//#define EXPECT_LE(x,y) CHECK((x) <= (y))
//#define EXPECT_GE(x,y) CHECK((x) >= (y))
//#define EXPECT_THROW(stmt, exc_type) CHECK_THROWS_AS( stmt, exc_type )

#include <gtest/gtest.h>

# include "../xsDNN/xsDNN.h"

# include "fully_connected_test.h"
# include "dropout_test.h"