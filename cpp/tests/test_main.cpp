/* 
 * Test framework for simple unit testing 
 * 
 * This is a minimalistic test framework that allows defining test cases with the TEST_CASE 
 * macro and performing assertions with CHECK and CHECK_CLOSE.
 * 
 * It uses static registration to collect test cases without needing an external framework.
 * 
 * The main() function runs all registered tests and reports pass/fail status, returning
 * a non-zero exit code if any tests fail.
 */

#include "test_framework.hpp"

#include <exception>
#include <iostream>

std::vector<TestCase>& test_registry() {
    // Static ensures the registry is created on first use and persists for the program lifetime.
    static std::vector<TestCase> tests;
    return tests;
}

TestRegistrar::TestRegistrar(const std::string& name, TestFn fn) {
    // Register the test case by adding it to the test registry.
    test_registry().push_back(TestCase{name, fn});
}

int main() {
    int failures = 0;
    for (const auto& test : test_registry()) {
        // Run each test and report std::exception failures without aborting the whole suite.
        try {
            test.fn();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& ex) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << ex.what() << '\n';
        }
    }
    // Return 0 on success, 1 if any test failed.
    return failures == 0 ? 0 : 1;
}
