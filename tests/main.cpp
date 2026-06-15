#include "nano_vdb.hpp"
#include <chrono>
#include <iostream>
#include <random>

int main() {
  // We use stable parameters to benchmark algorithmic differences
  const size_t DIMS = 128;
  NanoVectorDB db(DIMS);

  std::cout << "=== MILESTONE 4: NAVIGABLE GRAPH GENERATION ==="
            << std::endl;
  std::cout << "Inserting 10,000 stable vectors (building graph edges)...\n"
               "NOTE: Using optimized HNSW insertion (O(N log N)). It should be extremely fast!"
            << std::endl;

  // We use a fixed seed (42) to ensure reproducible tests across executions.
  // Using pure random numbers would result in fluctuating benchmark times.
  std::mt19937 engine(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Insert 10,000 vectors. During insertion, the 'insert' method
  // will create bidirectional connections (edges) between nodes to build the graph.
  for (int i = 0; i < 10000; ++i) {
    std::vector<float> random_vec(DIMS);
    for (size_t d = 0; d < DIMS; ++d) {
      random_vec[d] = dist(engine);
    }
    db.insert("node_" + std::to_string(i), random_vec);
  }

  // Generate a mock query for our speed test
  std::vector<float> query(DIMS);
  for (size_t d = 0; d < DIMS; ++d) {
    query[d] = dist(engine);
  }

  // =======================================================================
  // TEST 1: TRADITIONAL LINEAR SEARCH (Brute-Force)
  // Scans and calculates similarity against all 10,000 nodes.
  // =======================================================================
  std::cout << "\n--- TEST 1: LINEAR BRUTE-FORCE SEARCH (ALL NODES) ---"
            << std::endl;
  
  // Start benchmark timer
  auto start_lin = std::chrono::high_resolution_clock::now();
  auto res_linear = db.search_linear(query, 3);
  auto end_lin = std::chrono::high_resolution_clock::now();
  
  // Calculate elapsed microseconds
  auto time_linear =
      std::chrono::duration_cast<std::chrono::microseconds>(end_lin - start_lin)
          .count();

  for (const auto &res : res_linear) {
    std::cout << "ID: " << res.id << " | Score: " << res.score << std::endl;
  }
  std::cout << "Time elapsed: " << time_linear << " microseconds."
            << std::endl;


  // =======================================================================
  // TEST 2: GRAPH SEARCH (ANN - Approximate Nearest Neighbors)
  // Performs a Greedy Walk on the graph, exploring only a tiny fraction of nodes.
  // =======================================================================
  std::cout << "\n--- TEST 2: APPROXIMATE GRAPH SEARCH (GREEDY WALK) ---"
            << std::endl;
  
  // Start benchmark timer
  auto start_graph = std::chrono::high_resolution_clock::now();
  auto res_graph = db.search_graph(query, 3);
  auto end_graph = std::chrono::high_resolution_clock::now();
  
  auto time_graph = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_graph - start_graph)
                        .count();

  // PRINT RESULTS: Note! Because this is an "Approximate" algorithm,
  // results MAY differ slightly from Test 1.
  // In real-world LLMs, this tiny precision loss is an acceptable trade-off
  // for massive performance boosts (often 100x or 1000x faster).
  for (const auto &res : res_graph) {
    std::cout << "ID: " << res.id << " | Score: " << res.score << std::endl;
  }
  std::cout << "Time elapsed: " << time_graph << " microseconds."
            << std::endl;

  // Calculate speedup multiplier
  double speedup =
      static_cast<double>(time_linear) / static_cast<double>(time_graph);
  std::cout << "\n=== STRATEGIC RESULT ===" << std::endl;
  std::cout << "Graph Search is " << speedup
            << " times faster than full linear scanning." << std::endl;

  return 0;
}