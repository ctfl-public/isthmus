# Standalone GNU Make build for the surface-export demo without requiring CMake.

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
EXAMPLE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(EXAMPLE_DIR)/build
TARGET := $(BUILD_DIR)/isthmus_surface_export_demo
ISTHMUS_BUILD_DIR ?= $(ROOT_DIR)/build-wsl
ISTHMUS_LIB := $(ISTHMUS_BUILD_DIR)/libisthmus_cpp.a

# Keep the example Makefile simple: compile only the example translation unit
# and link against the prebuilt native ISTHMUS static library archive.
SOURCE := $(EXAMPLE_DIR)/main.cpp
OBJECT := $(BUILD_DIR)/main.o

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic
CPPFLAGS += -I$(ROOT_DIR)/cpp/include

.PHONY: all clean run check_isthmus_lib

all: check_isthmus_lib $(TARGET)

$(BUILD_DIR)/main.o: $(SOURCE) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJECT)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(ISTHMUS_LIB) $(LDLIBS) -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

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
