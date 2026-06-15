CXX     ?= c++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I./include

# Directories
SRC_DIR  = src
INC_DIR  = include
TEST_DIR = tests

# Files
SRC      = $(SRC_DIR)/nano_vdb.cpp
TEST_SRC = $(TEST_DIR)/main.cpp
TARGET   = flash_vdb_test

all: $(TARGET)

$(TARGET): $(SRC) $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET) snapshot.vdb

.PHONY: all clean
