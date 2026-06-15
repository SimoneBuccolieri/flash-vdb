#include "nano_vdb.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Calcola la Cosine Similarity tra il vettore query (float32) e il record
// quantizzato SQ8 (uint8_t), usando la formula algebrica:
//   dot(Q, D_original) = min_val * sum(Q) + scale * sum(Q_i * q_i)
// Il path NEON è compilato solo su arm64; altrimenti si usa il fallback scalare.
float NanoVectorDB::calculate_similarity(const std::vector<float> &query,
                                         float query_norm,
                                         float query_sum,
                                         const VectorRecord &record) const {
  float dot_product_q_qdata = 0.0f;

#ifdef __ARM_NEON
  // ── ARM NEON path (arm64) ───────────────────────────────────────────────
  // Processa 4 elementi per ciclo usando registri SIMD a 128-bit.
  size_t simd_loops = dimensions / 4;
  float32x4_t v_acc = vdupq_n_f32(0.0f);
  size_t i = 0;

  for (size_t j = 0; j < simd_loops; ++j) {
    // Carica 4 float della query in un registro SIMD
    float32x4_t q_vec = vld1q_f32(&query[i]);

    // Legge 4 byte consecutivi di qdata in un uint32_t (memcpy sicuro, no UB)
    uint32_t q_val;
    std::memcpy(&q_val, &record.qdata[i], sizeof(uint32_t));

    // Interpreta i 4 byte come uint8x8_t, estende a uint16, poi a uint32, poi a float
    uint8x8_t  d_u8  = vreinterpret_u8_u32(vdup_n_u32(q_val));
    uint16x8_t d_u16 = vmovl_u8(d_u8);
    uint32x4_t d_u32 = vmovl_u16(vget_low_u16(d_u16));
    float32x4_t d_f32 = vcvtq_f32_u32(d_u32);

    // Accumula: v_acc += q_vec * d_f32
    v_acc = vmlaq_f32(v_acc, q_vec, d_f32);
    i += 4;
  }

  // Riduce il vettore accumulatore a un singolo float
  dot_product_q_qdata = vaddvq_f32(v_acc);

  // Gestisce i rimanenti elementi se dimensions non è multiplo di 4
  for (; i < dimensions; ++i) {
    dot_product_q_qdata += query[i] * static_cast<float>(record.qdata[i]);
  }

#else
  // ── Scalar fallback (x86-64 e altri) ────────────────────────────────────
  for (size_t i = 0; i < dimensions; ++i) {
    dot_product_q_qdata += query[i] * static_cast<float>(record.qdata[i]);
  }
#endif

  // Ricostruisce il dot product effettivo tramite la formula algebrica SQ8
  float dot_product = record.min_val * query_sum + record.scale * dot_product_q_qdata;

  if (query_norm > 0.0f && record.norm > 0.0f) {
    return dot_product / (query_norm * record.norm);
  }
  return 0.0f;
}

// INSERIMENTO SUL GRAFO CON QUANTIZZAZIONE SQ8
bool NanoVectorDB::insert(const std::string &id,
                          const std::vector<float> &vec) {
  if (vec.size() != dimensions)
    return false;

  // --- COMPRESSIONE SQ8 PER-VETTORE ---
  // Troviamo il minimo, il massimo e la norma del vettore originale float.
  float min_val = vec[0];
  float max_val = vec[0];
  float sum_sq = 0.0f;
  for (float val : vec) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    sum_sq += val * val;
  }
  float norm = std::sqrt(sum_sq);

  // Calcoliamo la scala dinamica per mappare i valori float in [0, 255]
  float scale = (max_val - min_val) / 255.0f;
  if (scale < 1e-30f) {
    scale = 1.0f; // Previene divisione per zero se il vettore è uniforme
  }

  // Quantizziamo ogni coordinata float a uint8_t
  std::vector<uint8_t> qdata(dimensions);
  for (size_t i = 0; i < dimensions; ++i) {
    float q = std::round((vec[i] - min_val) / scale);
    if (q < 0.0f) q = 0.0f;
    if (q > 255.0f) q = 255.0f;
    qdata[i] = static_cast<uint8_t>(q);
  }

  // Creiamo il nuovo record quantizzato
  VectorRecord new_record;
  new_record.id = id;
  new_record.qdata = qdata;
  new_record.min_val = min_val;
  new_record.scale = scale;
  new_record.norm = norm;
  
  registry.push_back(new_record);
  size_t new_index = registry.size() - 1;

  // Se ci sono altri nodi nel database, colleghiamo il nuovo nodo ai vicini migliori
  if (registry.size() > 1) {
    // Usiamo search_graph per trovare i vicini più simili in O(log N)
    std::vector<SearchResult> best_neighbors = search_graph(vec, M);

    for (const SearchResult &res : best_neighbors) {
      size_t neighbor_idx = res.index;
      
      registry[new_index].neighbors.push_back(neighbor_idx);

      if (registry[neighbor_idx].neighbors.size() < M) {
        registry[neighbor_idx].neighbors.push_back(new_index);
      }
    }
  }
  return true;
}

// Funzione helper per ordinare i risultati di ricerca
bool compare_results(const SearchResult &a, const SearchResult &b) {
  return a.score > b.score;
}

// RICERCA LINEARE (BRUTE FORCE) CON QUANTIZZAZIONE SQ8 E SIMD
std::vector<SearchResult>
NanoVectorDB::search_linear(const std::vector<float> &query_vec,
                            size_t top_k) const {
  if (query_vec.size() != dimensions || registry.empty()) {
    std::vector<SearchResult> empty_res;
    return empty_res;
  }

  // Precalcoliamo le statistiche della query una sola volta per tutte le comparazioni
  float query_sum = 0.0f;
  float query_sum_sq = 0.0f;
  for (float val : query_vec) {
    query_sum += val;
    query_sum_sq += val * val;
  }
  float query_norm = std::sqrt(query_sum_sq);

  size_t actual_k = std::min(top_k, registry.size());
  std::vector<SearchResult> all_results;
  all_results.reserve(registry.size());

  for (size_t i = 0; i < registry.size(); ++i) {
    const VectorRecord &record = registry[i];
    float score = calculate_similarity(query_vec, query_norm, query_sum, record);
    
    SearchResult result;
    result.id = record.id;
    result.score = score;
    result.index = i;
    all_results.push_back(result);
  }

  std::partial_sort(all_results.begin(), all_results.begin() + actual_k,
                    all_results.end(), compare_results);

  std::vector<SearchResult> top_results;
  for (size_t i = 0; i < actual_k; ++i) {
    top_results.push_back(all_results[i]);
  }
  return top_results;
}

// RICERCA APPROSSIMATA SUL GRAFO (GREEDY WALK) CON SQ8 E SIMD
std::vector<SearchResult>
NanoVectorDB::search_graph(const std::vector<float> &query_vec,
                           size_t top_k) const {
  if (query_vec.size() != dimensions || registry.empty()) {
    std::vector<SearchResult> empty_res;
    return empty_res;
  }

  // Precalcoliamo le statistiche della query una sola volta
  float query_sum = 0.0f;
  float query_sum_sq = 0.0f;
  for (float val : query_vec) {
    query_sum += val;
    query_sum_sq += val * val;
  }
  float query_norm = std::sqrt(query_sum_sq);

  std::vector<bool> visited(registry.size(), false);
  std::vector<SearchResult> path_candidates;

  // Partiamo dal nodo fisso 0 (Entry Point)
  size_t current_idx = 0;
  float current_score =
      calculate_similarity(query_vec, query_norm, query_sum, registry[current_idx]);
  visited[current_idx] = true;

  SearchResult first_step;
  first_step.id = registry[current_idx].id;
  first_step.score = current_score;
  first_step.index = current_idx;
  path_candidates.push_back(first_step);

  bool keep_walking = true;
  while (keep_walking) {
    keep_walking = false;
    size_t best_neighbor_idx = current_idx;
    float best_neighbor_score = current_score;

    for (size_t neighbor_idx : registry[current_idx].neighbors) {
      if (!visited[neighbor_idx]) {
        visited[neighbor_idx] = true;
        
        float score = calculate_similarity(query_vec, query_norm, query_sum, registry[neighbor_idx]);
        
        SearchResult neighbor_candidate;
        neighbor_candidate.id = registry[neighbor_idx].id;
        neighbor_candidate.score = score;
        neighbor_candidate.index = neighbor_idx;
        path_candidates.push_back(neighbor_candidate);

        if (score > best_neighbor_score) {
          best_neighbor_score = score;
          best_neighbor_idx = neighbor_idx;
          keep_walking = true; // Abbiamo trovato un vicino più vicino, continuiamo a camminare!
        }
      }
    }

    if (keep_walking) {
      current_idx = best_neighbor_idx;
      current_score = best_neighbor_score;
    }
  }

  size_t actual_k = std::min(top_k, path_candidates.size());

  std::partial_sort(path_candidates.begin(), path_candidates.begin() + actual_k,
                    path_candidates.end(), compare_results);

  std::vector<SearchResult> top_results;
  for (size_t i = 0; i < actual_k; ++i) {
    top_results.push_back(path_candidates[i]);
  }
  return top_results;
}

// PERSISTENZA (Salvataggio su disco dei dati quantizzati)
bool NanoVectorDB::save_to_file(const std::string &filename) const {
  std::ofstream out(filename, std::ios::binary);
  if (!out)
    return false;

  out.write(reinterpret_cast<const char *>(&dimensions), sizeof(dimensions));
  size_t num_records = registry.size();
  out.write(reinterpret_cast<const char *>(&num_records), sizeof(num_records));

  for (const VectorRecord &record : registry) {
    size_t id_size = record.id.size();
    out.write(reinterpret_cast<const char *>(&id_size), sizeof(id_size));
    out.write(record.id.data(), id_size);

    // Salviamo qdata quantizzato (uint8_t) al posto di float a 32-bit
    out.write(reinterpret_cast<const char *>(record.qdata.data()),
              dimensions * sizeof(uint8_t));
    
    // Salviamo i parametri di dequantizzazione
    out.write(reinterpret_cast<const char *>(&record.min_val), sizeof(record.min_val));
    out.write(reinterpret_cast<const char *>(&record.scale), sizeof(record.scale));
    out.write(reinterpret_cast<const char *>(&record.norm), sizeof(record.norm));

    size_t num_neighbors = record.neighbors.size();
    out.write(reinterpret_cast<const char *>(&num_neighbors),
              sizeof(num_neighbors));
    if (num_neighbors > 0) {
      out.write(reinterpret_cast<const char *>(record.neighbors.data()),
                num_neighbors * sizeof(size_t));
    }
  }
  // Ritorna true solo se tutte le write sono andate a buon fine
  return out.good();
}

// Caricamento dei dati quantizzati e dei collegamenti del grafo da disco.
// Usa un vettore di staging locale: il registry viene aggiornato solo se il
// caricamento completo ha successo, preservando lo stato precedente in caso di errore.
bool NanoVectorDB::load_from_file(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) return false;

  size_t file_dims = 0;
  in.read(reinterpret_cast<char *>(&file_dims), sizeof(file_dims));
  if (in.fail() || file_dims != dimensions) return false;

  size_t num_records = 0;
  in.read(reinterpret_cast<char *>(&num_records), sizeof(num_records));
  if (in.fail()) return false;

  // Staging locale: non tocchiamo registry finché il load non è completato
  std::vector<VectorRecord> staging;
  staging.reserve(num_records);

  for (size_t i = 0; i < num_records; ++i) {
    size_t id_size = 0;
    in.read(reinterpret_cast<char *>(&id_size), sizeof(id_size));
    if (in.fail()) return false;

    std::string id(id_size, '\0');
    in.read(&id[0], id_size);
    if (in.fail()) return false;

    std::vector<uint8_t> qdata(dimensions);
    in.read(reinterpret_cast<char *>(qdata.data()), dimensions * sizeof(uint8_t));
    if (in.fail()) return false;

    float min_val = 0.0f, scale = 0.0f, norm = 0.0f;
    in.read(reinterpret_cast<char *>(&min_val), sizeof(min_val));
    in.read(reinterpret_cast<char *>(&scale), sizeof(scale));
    in.read(reinterpret_cast<char *>(&norm), sizeof(norm));
    if (in.fail()) return false;

    size_t num_neighbors = 0;
    in.read(reinterpret_cast<char *>(&num_neighbors), sizeof(num_neighbors));
    if (in.fail()) return false;

    std::vector<size_t> neighbors(num_neighbors);
    if (num_neighbors > 0) {
      in.read(reinterpret_cast<char *>(neighbors.data()),
              num_neighbors * sizeof(size_t));
      if (in.fail()) return false;
    }

    VectorRecord rec;
    rec.id        = std::move(id);
    rec.qdata     = std::move(qdata);
    rec.min_val   = min_val;
    rec.scale     = scale;
    rec.norm      = norm;
    rec.neighbors = std::move(neighbors);
    staging.push_back(std::move(rec));
  }

  // Commit atomico: il registry viene sostituito solo a caricamento riuscito
  registry = std::move(staging);
  return true;
}