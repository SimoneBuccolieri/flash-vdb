#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cerrno>

// POSIX socket headers
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// Request payload structure: matches the server's layout for zero text parsing.
struct QueryRequest {
    uint32_t top_k;
    float query_vec[128];
};

/**
 * Helper function to receive exactly 'size' bytes from a socket.
 * Handles partial reads (very common in TCP byte-stream channels) and EINTR interruptions.
 * Returns true if successful, false if connection closed or error occurred.
 */
bool recv_all(int socket_fd, char* buffer, size_t size) {
    size_t bytes_received = 0;
    while (bytes_received < size) {
        ssize_t result = recv(socket_fd, buffer + bytes_received, size - bytes_received, 0);
        if (result < 0) {
            if (errno == EINTR) continue; // System call was interrupted, retry
            std::cerr << "[Client] recv error: " << std::strerror(errno) << std::endl;
            return false;
        } else if (result == 0) {
            // Server closed connection
            return false;
        }
        bytes_received += result;
    }
    return true;
}

/**
 * Helper function to send exactly 'size' bytes to a socket.
 * Guarantees the complete transfer of a raw memory block despite TCP window limits.
 * Returns true if successful, false if send failed.
 */
bool send_all(int socket_fd, const char* buffer, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        ssize_t result = send(socket_fd, buffer + bytes_sent, size - bytes_sent, 0);
        if (result < 0) {
            if (errno == EINTR) continue; // System call was interrupted, retry
            std::cerr << "[Client] send error: " << std::strerror(errno) << std::endl;
            return false;
        }
        bytes_sent += result;
    }
    return true;
}

int main() {
    const size_t DIMS = 128;
    const int PORT = 8080;
    const char* SERVER_IP = "127.0.0.1";

    std::cout << "[Client] Preparing query request..." << std::endl;

    QueryRequest request;
    request.top_k = 3;

    // Build query vector matching "node_3" from the server to expect high similarity (Score ~ 1.0)
    // node_3 formula on server: vec[d] = (d + 30) * 0.01f
    for (size_t d = 0; d < DIMS; ++d) {
        request.query_vec[d] = static_cast<float>(d + 30) * 0.01f;
    }

    // =========================================================================
    // SOCKET LIFECYCLE: 1. socket()
    // =========================================================================
    // Creates an endpoint for communication.
    // AF_INET: IPv4 protocol family.
    // SOCK_STREAM: TCP stream sockets.
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        std::cerr << "[Client] Failed to create socket: " << std::strerror(errno) << std::endl;
        return 1;
    }

    // Prepare target server IPv4 address structure.
    struct sockaddr_in server_address;
    std::memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(PORT); // Convert port to network byte order

    // Converts IPv4 text address "127.0.0.1" into binary form.
    if (inet_pton(AF_INET, SERVER_IP, &server_address.sin_addr) <= 0) {
        std::cerr << "[Client] Invalid address or address format not supported: " << SERVER_IP << std::endl;
        close(sock_fd);
        return 1;
    }

    // =========================================================================
    // SOCKET LIFECYCLE: 2. connect()
    // =========================================================================
    // Initiates a connection on a socket to the specified server address.
    // Three-way handshake is performed synchronously.
    std::cout << "[Client] Connecting to " << SERVER_IP << ":" << PORT << "..." << std::endl;
    if (connect(sock_fd, reinterpret_cast<struct sockaddr*>(&server_address), sizeof(server_address)) < 0) {
        std::cerr << "[Client] Connection failed: " << std::strerror(errno) << std::endl;
        close(sock_fd);
        return 1;
    }
    std::cout << "[Client] Connected to server." << std::endl;

    // Transmit request structure directly in raw binary layout.
    std::cout << "[Client] Sending binary query vector (516 bytes)..." << std::endl;
    if (!send_all(sock_fd, reinterpret_cast<const char*>(&request), sizeof(QueryRequest))) {
        std::cerr << "[Client] Failed to send query request." << std::endl;
        close(sock_fd);
        return 1;
    }

    // Read the response from server.
    std::cout << "[Client] Waiting for response..." << std::endl;

    // 1. Read the number of results (uint32_t, 4 bytes)
    uint32_t num_results = 0;
    if (!recv_all(sock_fd, reinterpret_cast<char*>(&num_results), sizeof(num_results))) {
        std::cerr << "[Client] Failed to read response result size." << std::endl;
        close(sock_fd);
        return 1;
    }

    std::cout << "\n=== [Client] Search Results Received (" << num_results << ") ===" << std::endl;

    // 2. Loop and read each search result sequentially
    for (uint32_t i = 0; i < num_results; ++i) {
        // Read id length (uint32_t, 4 bytes)
        uint32_t id_len = 0;
        if (!recv_all(sock_fd, reinterpret_cast<char*>(&id_len), sizeof(id_len))) {
            std::cerr << "[Client] Failed to read result [" << i << "] ID length." << std::endl;
            break;
        }

        // Read the characters of the ID string
        std::string id(id_len, '\0');
        if (!recv_all(sock_fd, &id[0], id_len)) {
            std::cerr << "[Client] Failed to read result [" << i << "] ID value." << std::endl;
            break;
        }

        // Read the float score (4 bytes)
        float score = 0.0f;
        if (!recv_all(sock_fd, reinterpret_cast<char*>(&score), sizeof(score))) {
            std::cerr << "[Client] Failed to read result [" << i << "] score." << std::endl;
            break;
        }

        // Output results to stdout. No text manipulation was performed in transit!
        std::cout << "Rank " << (i + 1) << " | Node ID: " << id << " | Cosine Similarity Score: " << score << std::endl;
    }

    std::cout << "=============================================\n" << std::endl;

    // Close the connection
    std::cout << "[Client] Closing socket connection." << std::endl;
    close(sock_fd);

    return 0;
}
