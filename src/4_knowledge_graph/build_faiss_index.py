import faiss
import numpy as np
import json
import os
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
EMBEDDINGS_FILE = "data/embeddings/document_embeddings.npy"
METADATA_FILE = "data/classified/classified_documents.json"
OUTPUT_DIR = "models/ranker/faiss"

# FAISS Parameters
HNSW_M = 32  # Number of connections per layer
HNSW_EF_CONSTRUCTION = 200  # Higher = better quality, slower build
HNSW_EF_SEARCH = 64  # Higher = better recall, slower search


def load_data():
    """Load embeddings and metadata"""
    logger.info(f"Loading embeddings from: {EMBEDDINGS_FILE}")
    
    if not os.path.exists(EMBEDDINGS_FILE):
        logger.error(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        return None, None
    
    embeddings = np.load(EMBEDDINGS_FILE)
    logger.info(f"✓ Loaded {embeddings.shape[0]} embeddings (dim: {embeddings.shape[1]})")
    
    # Load metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    logger.info(f"✓ Loaded metadata for {len(documents)} documents")
    
    return embeddings, documents


def build_flat_index(embeddings):
    """Build flat (exact) index"""
    logger.info("\n[Building] Flat Index (Exact Search)...")
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine for normalized vectors
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add vectors
    index.add(embeddings)
    logger.info(f"   ✓ Added {index.ntotal} vectors")
    logger.info(f"   ✓ Index type: Flat (exact)")
    
    return index


def build_hnsw_index(embeddings):
    """Build HNSW index for fast approximate search"""
    logger.info("\n[Building] HNSW Index (Graph-based Search)...")
    logger.info(f"   M (connections): {HNSW_M}")
    logger.info(f"   efConstruction: {HNSW_EF_CONSTRUCTION}")
    
    dim = embeddings.shape[1]
    
    # Create HNSW index
    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings)
    
    # Add vectors
    logger.info("   Adding vectors...")
    index.add(embeddings)
    
    # Set search parameters
    index.hnsw.efSearch = HNSW_EF_SEARCH
    
    logger.info(f"   ✓ Added {index.ntotal} vectors")
    logger.info(f"   ✓ Index type: HNSW{HNSW_M}")
    logger.info(f"   ✓ efSearch: {HNSW_EF_SEARCH}")
    
    return index


def test_index(index, embeddings, k=10, num_queries=100):
    """Test index performance"""
    logger.info("\n[Testing] Index Performance...")
    logger.info(f"   Queries: {num_queries}")
    logger.info(f"   Top-K: {k}")
    
    # Random query vectors
    query_indices = np.random.choice(embeddings.shape[0], num_queries, replace=False)
    queries = embeddings[query_indices].copy()
    faiss.normalize_L2(queries)
    
    # Search
    start = time.time()
    distances, indices = index.search(queries, k)
    elapsed = time.time() - start
    
    # Stats
    logger.info(f"\n   Results:")
    logger.info(f"   ✓ Total time: {elapsed:.3f}s")
    logger.info(f"   ✓ Queries/second: {num_queries/elapsed:.1f}")
    logger.info(f"   ✓ Latency per query: {elapsed/num_queries*1000:.2f}ms")
    logger.info(f"   ✓ Average distance to top-1: {distances[:, 0].mean():.4f}")
    
    return distances, indices


def save_index(index, index_type, embeddings, documents):
    """Save FAISS index and metadata"""
    logger.info(f"\n[Saving] FAISS Index...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save index
    index_path = os.path.join(OUTPUT_DIR, f"{index_type}_index.faiss")
    faiss.write_index(index, index_path)
    logger.info(f"   ✓ Index saved: {index_path}")
    
    # Create metadata
    metadata = {
        'index_type': index_type,
        'num_vectors': int(index.ntotal),
        'dimension': int(embeddings.shape[1]),
        'metric': 'inner_product',
        'normalized': True,
        'created_at': datetime.now().isoformat(),
        'index_params': {}
    }
    
    # Add index-specific parameters
    if index_type == 'hnsw':
        # Access HNSW parameters correctly
        try:
            # Try to get M from the index structure
            # FAISS stores M in the HNSW structure but access varies by version
            if hasattr(index, 'hnsw'):
                # For newer FAISS versions
                metadata['index_params'] = {
                    'M': HNSW_M,  # Use the constant we defined
                    'efConstruction': HNSW_EF_CONSTRUCTION,
                    'efSearch': HNSW_EF_SEARCH
                }
            else:
                metadata['index_params'] = {
                    'M': HNSW_M,
                    'efConstruction': HNSW_EF_CONSTRUCTION,
                    'efSearch': HNSW_EF_SEARCH
                }
        except Exception as e:
            logger.warning(f"   Could not extract HNSW params: {e}")
            metadata['index_params'] = {
                'M': HNSW_M,
                'efConstruction': HNSW_EF_CONSTRUCTION,
                'efSearch': HNSW_EF_SEARCH
            }
    
    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, f"{index_type}_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   ✓ Metadata saved: {meta_path}")
    
    # Save document mapping
    mapping = {
        'total_documents': len(documents),
        'documents': []
    }
    
    for idx, doc in enumerate(documents):
        doc_info = {
            'index': idx,
            'title': doc.get('title', ''),
            'heritage_types': doc.get('classifications', {}).get('heritage_types', []),
            'domains': doc.get('classifications', {}).get('domains', []),
            'cluster_label': doc.get('cluster_label', ''),
            'source': doc.get('source', '')
        }
        mapping['documents'].append(doc_info)
    
    mapping_path = os.path.join(OUTPUT_DIR, "document_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"   ✓ Mapping saved: {mapping_path}")


def compare_indices(flat_index, hnsw_index, embeddings, k=10):
    """Compare flat vs HNSW index accuracy"""
    logger.info("\n[Comparison] Flat vs HNSW...")
    
    num_test = 100
    query_indices = np.random.choice(embeddings.shape[0], num_test, replace=False)
    queries = embeddings[query_indices].copy()
    faiss.normalize_L2(queries)
    
    # Get results from both indices
    _, flat_results = flat_index.search(queries, k)
    _, hnsw_results = hnsw_index.search(queries, k)
    
    # Calculate recall@k
    recalls = []
    for i in range(num_test):
        flat_set = set(flat_results[i])
        hnsw_set = set(hnsw_results[i])
        overlap = len(flat_set & hnsw_set)
        recall = overlap / k
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    logger.info(f"   HNSW Recall@{k}: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    logger.info(f"   (How many of HNSW's results match exact search)")


def main():
    logger.info("="*70)
    logger.info("FAISS INDEX BUILDER")
    logger.info("="*70)
    
    # Load data
    embeddings, documents = load_data()
    if embeddings is None:
        return
    
    # Determine best index type
    n_docs = embeddings.shape[0]
    logger.info(f"\n[Configuration]")
    logger.info(f"   Dataset size: {n_docs} documents")
    logger.info(f"   Embedding dim: {embeddings.shape[1]}")
    
    if n_docs < 10000:
        logger.info(f"   Recommended: Flat (exact) index")
    else:
        logger.info(f"   Recommended: HNSW (approximate) index")
    
    # Build flat index (always, for exact search)
    flat_index = build_flat_index(embeddings.copy())
    
    # Test flat index
    test_index(flat_index, embeddings)
    
    # Save flat index
    save_index(flat_index, 'flat', embeddings, documents)
    
    # Build HNSW index for comparison
    logger.info("\n[Bonus] Building additional index types for comparison...")
    hnsw_index = build_hnsw_index(embeddings.copy())
    
    # Save HNSW index
    save_index(hnsw_index, 'hnsw', embeddings, documents)
    
    # Test HNSW index
    test_index(hnsw_index, embeddings)
    
    # Compare accuracy
    compare_indices(flat_index, hnsw_index, embeddings)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("FAISS INDEX BUILD COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Built indices for {n_docs} documents")
    logger.info(f"📊 Index types:")
    logger.info(f"   - Flat (exact): For best accuracy")
    logger.info(f"   - HNSW (approx): For fast search")
    logger.info(f"\n💾 Files created in: {OUTPUT_DIR}")
    logger.info(f"   - flat_index.faiss")
    logger.info(f"   - hnsw_index.faiss")
    logger.info(f"   - flat_metadata.json")
    logger.info(f"   - hnsw_metadata.json")
    logger.info(f"   - document_mapping.json")
    logger.info("\n💡 Usage:")
    logger.info("   Use Flat index for production (exact results)")
    logger.info("   Use HNSW index if you need faster search (slight accuracy tradeoff)")
    logger.info("="*70)


if __name__ == "__main__":
    main()