#include "nano_vdb.hpp"
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

// Request payload structure: fixed-size layout for zero text parsing.
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
            std::cerr << "[Server] recv error: " << std::strerror(errno) << std::endl;
            return false;
        } else if (result == 0) {
            // Client closed the connection orderly
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
            std::cerr << "[Server] send error: " << std::strerror(errno) << std::endl;
            return false;
        }
        bytes_sent += result;
    }
    return true;
}

int main() {
    const size_t DIMS = 128;
    const int PORT = 8080;

    std::cout << "[Server] Initializing FlashVDB Database with " << DIMS << " dimensions..." << std::endl;
    NanoVectorDB db(DIMS);

    // Populate database with 5 distinct mock vectors.
    // Each vector has progressive float values to test cosine similarity differences.
    std::cout << "[Server] Populating database with mock vectors..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::vector<float> vec(DIMS);
        // Fill vector with distinct float patterns
        for (size_t d = 0; d < DIMS; ++d) {
            vec[d] = static_cast<float>(d + i * 10) * 0.01f;
        }
        std::string id = "node_" + std::to_string(i);
        if (db.insert(id, vec)) {
            std::cout << "  -> Inserted " << id << " successfully." << std::endl;
        } else {
            std::cerr << "  -> Failed to insert " << id << std::endl;
        }
    }
    std::cout << "[Server] Database populated. Size: " << db.size() << " records." << std::endl;

    // =========================================================================
    // SOCKET LIFECYCLE: 1. socket()
    // =========================================================================
    // AF_INET: IPv4 Internet protocols.
    // SOCK_STREAM: Provides sequenced, reliable, two-way, connection-based byte streams (TCP).
    // 0: Protocol choice (IP).
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "[Server] Failed to create socket: " << std::strerror(errno) << std::endl;
        return 1;
    }

    // Set SO_REUSEADDR option so we can restart the server immediately without waiting for TCP TIME_WAIT to clear.
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "[Server] setsockopt SO_REUSEADDR failed: " << std::strerror(errno) << std::endl;
        close(server_fd);
        return 1;
    }

    // =========================================================================
    // SOCKET LIFECYCLE: 2. bind()
    // =========================================================================
    // Assigns the local protocol address to the socket.
    // In IPv4 (sockaddr_in), this includes the IP address and the port number.
    struct sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY; // Listen on all network interfaces
    address.sin_port = htons(PORT);       // Convert port number to network byte order (Big Endian)

    if (bind(server_fd, reinterpret_cast<struct sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "[Server] bind failed on port " << PORT << ": " << std::strerror(errno) << std::endl;
        close(server_fd);
        return 1;
    }

    // =========================================================================
    // SOCKET LIFECYCLE: 3. listen()
    // =========================================================================
    // Puts the socket in passive listening state, ready to accept incoming connections.
    // Backlog (10): maximum length of the queue of pending connections.
    if (listen(server_fd, 10) < 0) {
        std::cerr << "[Server] listen failed: " << std::strerror(errno) << std::endl;
        close(server_fd);
        return 1;
    }

    std::cout << "[Server] Listening on port " << PORT << " (TCP)... Waiting for client connection." << std::endl;

    while (true) {
        // =====================================================================
        // SOCKET LIFECYCLE: 4. accept()
        // =====================================================================
        // Blocks the execution until a client connection arrives.
        // It extracts the first connection request on the queue of pending connections,
        // creates a new active socket, and returns a new file descriptor referring to that socket.
        struct sockaddr_in client_address;
        socklen_t client_addr_len = sizeof(client_address);
        int client_fd = accept(server_fd, reinterpret_cast<struct sockaddr*>(&client_address), &client_addr_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue; // Interrupted by signal handler, keep waiting
            std::cerr << "[Server] accept failed: " << std::strerror(errno) << std::endl;
            continue;
        }

        // Print client connection information
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_address.sin_addr, client_ip, INET_ADDRSTRLEN);
        std::cout << "[Server] Client connected from " << client_ip << ":" << ntohs(client_address.sin_port) << std::endl;

        // Read request header & query vector (Zero Text Parsing)
        QueryRequest request;
        if (!recv_all(client_fd, reinterpret_cast<char*>(&request), sizeof(QueryRequest))) {
            std::cerr << "[Server] Failed to receive full request from client." << std::endl;
            close(client_fd);
            continue;
        }

        std::cout << "[Server] Query received: top_k = " << request.top_k << ". Performing graph search..." << std::endl;

        // Convert query float array to std::vector<float>
        std::vector<float> query_vec(request.query_vec, request.query_vec + DIMS);

        // Execute graph search (ANN)
        std::vector<SearchResult> results = db.search_graph(query_vec, request.top_k);

        // =====================================================================
        // RESPONSE SERIALIZATION (ZERO TEXT PARSING, CONTINUOUS MEMORY)
        // =====================================================================
        // Pack all values contiguously into a binary stream.
        // Format: [num_results (uint32_t)] + [ [id_len (uint32_t)][id_chars (char*)][score (float)] ] x num_results
        std::vector<char> response_buffer;

        uint32_t num_results = static_cast<uint32_t>(results.size());
        const char* num_ptr = reinterpret_cast<const char*>(&num_results);
        response_buffer.insert(response_buffer.end(), num_ptr, num_ptr + sizeof(num_results));

        for (const auto& res : results) {
            // 1. Serialize the length of ID string
            uint32_t id_len = static_cast<uint32_t>(res.id.size());
            const char* len_ptr = reinterpret_cast<const char*>(&id_len);
            response_buffer.insert(response_buffer.end(), len_ptr, len_ptr + sizeof(id_len));

            // 2. Serialize raw ID characters (no null-terminator sent, client uses length)
            response_buffer.insert(response_buffer.end(), res.id.begin(), res.id.end());

            // 3. Serialize score (float, 4 bytes)
            const char* score_ptr = reinterpret_cast<const char*>(&res.score);
            response_buffer.insert(response_buffer.end(), score_ptr, score_ptr + sizeof(res.score));
        }

        // Send the entire packed buffer in one single syscall to optimize bandwidth and latency
        if (!send_all(client_fd, response_buffer.data(), response_buffer.size())) {
            std::cerr << "[Server] Failed to transmit search results to client." << std::endl;
        } else {
            std::cout << "[Server] Search completed. Sent " << num_results << " results to client." << std::endl;
        }

        // Close connection with current client
        close(client_fd);
    }

    close(server_fd);
    return 0;
}
