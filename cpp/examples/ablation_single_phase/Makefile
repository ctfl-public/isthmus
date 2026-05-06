# Standalone GNU Make build for the single-phase ablation demo

# Root of the ISTHMUS++ repository
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
# Directory of this example
EXAMPLE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# Output directory for compiled object and executable files
BUILD_DIR := $(EXAMPLE_DIR)/build
# Final executable
TARGET := $(BUILD_DIR)/ablation_demo
# Where the main ISTHMUS library was built
ISTHMUS_BUILD_DIR ?= $(ROOT_DIR)/build-wsl
# ISTHMUS_LIB is the path to the prebuilt static library archive we will link against.
ISTHMUS_LIB := $(ISTHMUS_BUILD_DIR)/libisthmus_cpp.a

SOURCE := $(EXAMPLE_DIR)/main.cpp
OBJECT := $(BUILD_DIR)/main.o

# Use g++ by default, but allow override via environment variable
CXX ?= g++ 
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic
# Where to find the ISTHMUS headers
CPPFLAGS += -I$(ROOT_DIR)/cpp/include

.PHONY: all clean run check_isthmus_lib

all: check_isthmus_lib $(TARGET)

# Compile source files into object files, ensuring the build directory exists first
$(BUILD_DIR)/main.o: $(SOURCE) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Link object files into the final executable, 
# linking against the ISTHMUS library and any necessary system libraries
$(TARGET): $(OBJECT)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(ISTHMUS_LIB) $(LDLIBS) -o $@

# Ensure the build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Check that the ISTHMUS static library exists
check_isthmus_lib:
	@test -f "$(ISTHMUS_LIB)" || { \
		echo "Missing ISTHMUS static library: $(ISTHMUS_LIB)"; \
		echo "Build the main library first, or override ISTHMUS_BUILD_DIR."; \
		exit 1; \
	}

run: $(TARGET)
	$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
