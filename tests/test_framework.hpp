#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// create a type alias for test function pointer, 
//  a pointer to a function that takes no arguments and returns void
using TestFn = void(*)();

// A simple struct to hold test case information, 
// including the test name and the function to execute.
struct TestCase {
    std::string name;
    TestFn fn;
};

// test_registry() function returns a reference to a static vector of TestCase objects, 
//  which serves as the central registry for all test cases defined in the program. 
std::vector<TestCase>& test_registry();

// TestRegistrar struct with a constructor that registers a test case by adding it to the test registry.
struct TestRegistrar {
    TestRegistrar(const std::string& name, TestFn fn);
};

// Macros to define test cases and perform assertions.

/* This allows users to write TEST_CASE(test_name) { ... } to define a test case, and it will be automatically registered.
 * 
 * The macro works as follows:  
 * TEST_CASE(my_test) {
 *     CHECK(1 == 1);
 * }
 * 
 * // Expands to:
 * void my_test();
 * static TestRegistrar registrar_my_test("my_test", &my_test);  // Created at startup!
 * void my_test() {
 *     CHECK(1 == 1);
 * }
 * 
 * When the program loads, registrar_my_test is created, its constructor runs, and it registers the test before main() even starts.
 * 
 * The macro works as follows:
 *  First, it declares a function named 'name' that takes no arguments and returns void.
 *  Then, it creates a static instance of TestRegistrar named 'registrar_name' that registers 
 *      the test case with the provided name and function pointer.
 *  Finally, it defines the function 'name'
 * 
 * The #name syntax converts the argument 'name' into a string literal, which is used for registration.
 * The static instance of TestRegistrar ensures that the test case is registered before main() is executed,
 *  allowing the test framework to discover all test cases automatically.
 * 
 */
#define TEST_CASE(name) \
    void name(); \
    static TestRegistrar registrar_##name(#name, &name); \
    void name()

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error("CHECK failed: " #cond); \
        } \
    } while (false)

#define CHECK_CLOSE(a, b, eps) \
    do { \
        const auto lhs_ = (a); \
        const auto rhs_ = (b); \
        if (std::abs(lhs_ - rhs_) > (eps)) { \
            throw std::runtime_error("CHECK_CLOSE failed"); \
        } \
    } while (false)
