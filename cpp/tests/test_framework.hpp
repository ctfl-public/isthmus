#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using TestFn = void(*)();

struct TestCase {
    std::string name;
    TestFn fn;
};

std::vector<TestCase>& test_registry();

// Static registration keeps tests lightweight without bringing in an external framework.
struct TestRegistrar {
    TestRegistrar(const std::string& name, TestFn fn);
};

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
