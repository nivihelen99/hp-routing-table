# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

# Sources and targets
TEST_SRC := gtest.cpp
TEST_TARGET := test_runner
EXAMPLE_SRC := example.cpp
EXAMPLE_TARGET := example_app

# Google Test options
GTEST_LIBS := -lgtest -lgtest_main -pthread

# Default target: build both
all: $(TEST_TARGET) $(EXAMPLE_TARGET)

# Build Google Test target
$(TEST_TARGET): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(GTEST_LIBS)

# Build standalone example target
$(EXAMPLE_TARGET): $(EXAMPLE_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Run the Google Test binary
run-test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Run the example binary
run-example: $(EXAMPLE_TARGET)
	./$(EXAMPLE_TARGET)

# Clean both targets
clean:
	rm -f $(TEST_TARGET) $(EXAMPLE_TARGET)
