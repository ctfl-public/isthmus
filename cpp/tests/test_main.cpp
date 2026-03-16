#include "test_framework.hpp"

#include <exception>
#include <iostream>

std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> tests;
    return tests;
}

TestRegistrar::TestRegistrar(const std::string& name, TestFn fn) {
    test_registry().push_back(TestCase{name, fn});
}

int main() {
    int failures = 0;
    for (const auto& test : test_registry()) {
        try {
            test.fn();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& ex) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << ex.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}
