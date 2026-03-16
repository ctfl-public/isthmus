#pragma once

#include <stdexcept>
#include <string>

namespace isthmus {

/*
 * Base exception type for the native library.
 *
 * Callers that want one catch point for all ISTHMUS-native failures can catch
 * this type. More specific exception classes below communicate whether the
 * problem came from bad input or from requesting an algorithm stage that has
 * not yet been implemented.
 */
class IsthmusError : public std::runtime_error {
public:
    explicit IsthmusError(const std::string& message)
        : std::runtime_error(message) {}
};

/*
 * Thrown when the caller provides a domain, voxel size, or voxel set that the
 * marching-windows algorithm cannot interpret safely.
 */
class InvalidInputError : public IsthmusError {
public:
    explicit InvalidInputError(const std::string& message)
        : IsthmusError(message) {}
};

/*
 * Thrown when the public API is asked to execute a stage whose native backend
 * has not been added yet.
 */
class NotImplementedError : public IsthmusError {
public:
    explicit NotImplementedError(const std::string& message)
        : IsthmusError(message) {}
};

}  // namespace isthmus
