# ⚡ FlashVDB

A zero-dependency, SIMD-accelerated in-memory vector database designed for Edge AI and high-performance local execution.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-informational.svg)](https://en.cppreference.com/w/cpp/17)
[![SIMD](https://img.shields.io/badge/SIMD-NEON%20%7C%20AVX2-orange.svg)]()

---

## The Problem

Modern vector databases are built for cloud-scale infrastructure, dragging along massive dependencies, HTTP overhead, and heavy memory footprints. When building Edge AI applications or local LLM tools, you don't need a distributed cluster; you need raw, bare-metal performance.

FlashVDB is a lightweight C++ engine that acts like SQLite for vector embeddings. It relies on static memory allocation, hardware-level parallelization, and zero-copy persistence to deliver maximum throughput on local machines.

---

## Core Features

- **Zero Dependencies:** Written in modern C++17. No external libraries, no bloated frameworks.
- **Hardware Acceleration:** Native SIMD optimization (ARM NEON & AVX2 support) to compute vector similarities in parallel across 128-bit/256-bit CPU registers.
- **Raw Binary Persistence:** Disk I/O is handled via direct memory dumping (Redis RDB style). No string parsing (JSON/CSV), enabling multi-gigabyte index loads in milliseconds.
- **Approximate Graph Search:** Implements a Greedy Walk algorithm with an exploration frontier (`efSearch`) to bypass local minimums and achieve $O(\log N)$ search complexity.

---

## Quickstart

Build the test suite with the included Makefile (requires `g++` or `clang++`):

```bash
make
./flash_vdb_test
```

### Basic Usage API

```cpp
#include "nano_vdb.hpp"

// Initialize database with 128 dimensions
NanoVectorDB db(128);

// Insert vectors (graph index is built on the fly)
db.insert("doc_1", {0.12f, 0.88f, /* ... */});

// Search top 3 most similar vectors
std::vector<float> query = {0.10f, 0.85f, /* ... */};
auto results = db.search_graph(query, 3);

// Persist instantly to disk
db.save_to_file("snapshot.vdb");
```

---

## Architecture Notes

The distance metric used is Cosine Similarity. Memory is kept strictly contiguous (`std::vector<float>`) to maximize CPU cache-line hits. The search algorithm maintains an internal priority queue to balance latency and recall.

---

## Repository Layout

```
flash-vdb/
├── include/
│   └── nano_vdb.hpp     # Public API header
├── src/
│   └── nano_vdb.cpp     # Core implementation
├── tests/
│   └── main.cpp         # Test & benchmark entry point
├── docs/                # Extended documentation
├── Makefile
└── README.md
```

---

## License

MIT. See [LICENSE](LICENSE).
