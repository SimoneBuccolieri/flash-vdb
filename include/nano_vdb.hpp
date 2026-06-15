#ifndef NANO_VDB_HPP
#define NANO_VDB_HPP

#include <string>
#include <vector>
#include <cstdint>

// Represents a single item in the database.
// This is effectively a "Node" in our graph representation.
struct VectorRecord {
  std::string id; // Unique identifier for the document (e.g., "doc_123")
  std::vector<uint8_t> qdata; // Quantized embedding vector (SQ8)
  float min_val; // Minimum value of original vector dimensions (for dequantization)
  float scale;   // Quantization scale (for dequantization)
  float norm;    // Precomputed norm of the original vector

  // NEW: Stores the indices of neighboring nodes in the graph.
  // Instead of searching the entire DB, we will hop from neighbor to neighbor.
  // This is the core concept behind ANN (Approximate Nearest Neighbors)
  // algorithms like HNSW!
  std::vector<size_t> neighbors;
};

// Represents a search result (ID + Similarity Score)
struct SearchResult {
  std::string id;
  float score;
  size_t index; // The internal array index of the node
};

// Main class managing the Vector Database
class NanoVectorDB {
private:
  std::vector<VectorRecord> registry; // In-memory storage containing all nodes
  size_t dimensions; // Fixed length for all vectors (e.g., 128)

  // Maximum number of connections (edges) per node.
  // Limiting this to 4 keeps the graph lightweight to traverse
  // while ensuring it remains a "Small World Graph".
  const size_t M = 4;

  // Internal function to calculate Cosine Similarity between two vectors.
  // Returns a score from -1 (opposite) to 1 (identical).
  float calculate_similarity(const std::vector<float> &query,
                             float query_norm,
                             float query_sum,
                             const VectorRecord &record) const;

public:
  explicit NanoVectorDB(size_t dims) : dimensions(dims) {}

  // Insertion now builds the graph: it finds the 'M' closest existing nodes
  // and creates bidirectional edges between them and the new node.
  bool insert(const std::string &id, const std::vector<float> &vec);

  // Graph Search (ANN). Instead of checking all nodes O(N), it performs
  // a Greedy Walk starting from an entry point and moving towards the best
  // neighbor. This reduces search complexity to approximately O(log N).
  std::vector<SearchResult> search_graph(const std::vector<float> &query_vec,
                                         size_t top_k) const;

  // Traditional Linear Search (Brute Force) that checks ALL nodes.
  // Slower, but mathematically guarantees 100% exact matches.
  // Highly useful as a baseline for benchmarking against search_graph.
  std::vector<SearchResult> search_linear(const std::vector<float> &query_vec,
                                          size_t top_k) const;

  // Saves the entire database (vectors + graph edges) to a binary file
  bool save_to_file(const std::string &filename) const;

  // Loads the database from a binary file into memory
  bool load_from_file(const std::string &filename);

  size_t size() const { return registry.size(); }
};

#endif