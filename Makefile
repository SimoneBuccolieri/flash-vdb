CXX     ?= c++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I./include

# Directories
SRC_DIR  = src
INC_DIR  = include
TEST_DIR = tests
NET_DIR  = src/network

# Files
SRC         = $(SRC_DIR)/nano_vdb.cpp
TEST_SRC    = $(TEST_DIR)/main.cpp
SERVER_SRC  = $(NET_DIR)/server_node.cpp
CLIENT_SRC  = $(NET_DIR)/client_node.cpp

# Targets
TARGET_TEST   = flash_vdb_test
TARGET_SERVER = flash_server
TARGET_CLIENT = flash_client

all: $(TARGET_TEST) $(TARGET_SERVER) $(TARGET_CLIENT)

$(TARGET_TEST): $(SRC) $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TARGET_SERVER): $(SRC) $(SERVER_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TARGET_CLIENT): $(CLIENT_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET_TEST) $(TARGET_SERVER) $(TARGET_CLIENT) snapshot.vdb

.PHONY: all clean
