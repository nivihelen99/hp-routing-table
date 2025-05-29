# Makefile for gtest.cpp using Google Test

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

# Paths
SRC := gtest.cpp
TARGET := test_runner

# Google Test options (assumes installed via package manager or system-wide)
GTEST_LIBS := -lgtest -lgtest_main -pthread

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(GTEST_LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
