import pickle
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime
import os
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# File paths
KG_FILE = "knowledge_graph/heritage_kg.gpickle"
SIMRANK_DIR = "knowledge_graph/simrank"
SIMRANK_MATRIX_FILE = os.path.join(SIMRANK_DIR, "simrank_matrix.npy")
SIMRANK_MAPPING_FILE = os.path.join(SIMRANK_DIR, "simrank_mapping.json")

# SimRank parameters
MAX_ITERATIONS = 10
DECAY_FACTOR = 0.8  # C parameter
CONVERGENCE_THRESHOLD = 0.001


def load_knowledge_graph():
    """Load the knowledge graph"""
    logger.info(f"Loading knowledge graph from: {KG_FILE}")
    
    if not os.path.exists(KG_FILE):
        logger.error(f"Knowledge graph file not found: {KG_FILE}")
        logger.error("Please run 5_build_knowledge_graph.py first!")
        return None
    
    with open(KG_FILE, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"✓ Loaded KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Get document nodes only
    doc_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'document']
    logger.info(f"✓ Found {len(doc_nodes)} document nodes")
    
    return G, doc_nodes


def compute_simrank_optimized(G, doc_nodes, max_iter=MAX_ITERATIONS, 
                              decay=DECAY_FACTOR, threshold=CONVERGENCE_THRESHOLD):
    """
    Optimized SimRank computation for undirected graphs
    """
    logger.info("\n[SimRank] Configuration:")
    logger.info(f"   Max iterations: {max_iter}")
    logger.info(f"   Decay factor (C): {decay}")
    logger.info(f"   Convergence threshold: {threshold}")
    logger.info(f"   Document nodes: {len(doc_nodes)}")
    
    n = len(doc_nodes)
    
    # Create node to index mapping
    node_to_idx = {node: idx for idx, node in enumerate(doc_nodes)}
    
    # Initialize similarity matrix (diagonal = 1, rest = 0)
    sim_matrix = np.eye(n, dtype=np.float32)
    
    # Build neighbor index for faster lookup
    logger.info("\n[Phase 1] Building neighbor index...")
    neighbor_index = {}
    
    for node in tqdm(doc_nodes, desc="Indexing neighbors"):
        # For undirected graphs, use G.neighbors()
        neighbors = list(G.neighbors(node))
        neighbor_index[node] = neighbors
    
    logger.info(f"✓ Indexed neighbors for {len(neighbor_index)} nodes")
    
    # Iterative SimRank computation
    logger.info("\n[Phase 2] Computing SimRank similarities...")
    
    for iteration in range(max_iter):
        logger.info(f"\nIteration {iteration + 1}/{max_iter}")
        
        old_sim_matrix = sim_matrix.copy()
        
        # Update similarity for all pairs
        for i in tqdm(range(n), desc=f"Iter {iteration+1}"):
            node_i = doc_nodes[i]
            neighbors_i = neighbor_index[node_i]
            
            if len(neighbors_i) == 0:
                continue
            
            for j in range(i + 1, n):
                node_j = doc_nodes[j]
                neighbors_j = neighbor_index[node_j]
                
                if len(neighbors_j) == 0:
                    continue
                
                # SimRank formula: C * avg(sim(neighbors))
                sim_sum = 0.0
                count = 0
                
                for ni in neighbors_i:
                    if ni in node_to_idx:
                        ni_idx = node_to_idx[ni]
                        for nj in neighbors_j:
                            if nj in node_to_idx:
                                nj_idx = node_to_idx[nj]
                                sim_sum += old_sim_matrix[ni_idx, nj_idx]
                                count += 1
                
                if count > 0:
                    new_sim = decay * (sim_sum / count)
                    sim_matrix[i, j] = new_sim
                    sim_matrix[j, i] = new_sim  # Symmetric
        
        # Check convergence
        diff = np.abs(sim_matrix - old_sim_matrix).max()
        logger.info(f"   Max change: {diff:.6f}")
        
        if diff < threshold:
            logger.info(f"   ✓ Converged at iteration {iteration + 1}")
            break
    
    logger.info(f"\n✓ SimRank computation complete")
    logger.info(f"   Final matrix shape: {sim_matrix.shape}")
    logger.info(f"   Non-zero similarities: {np.count_nonzero(sim_matrix > 0.01)}")
    
    return sim_matrix, node_to_idx


def analyze_simrank_results(sim_matrix, doc_nodes, G):
    """Analyze SimRank results"""
    logger.info("\n[Phase 3] Analyzing SimRank results...")
    
    n = len(doc_nodes)
    
    # Basic statistics
    non_diag_mask = ~np.eye(n, dtype=bool)
    non_diag_sims = sim_matrix[non_diag_mask]
    
    stats = {
        'total_pairs': int(n * (n - 1) / 2),
        'mean_similarity': float(non_diag_sims.mean()),
        'std_similarity': float(non_diag_sims.std()),
        'min_similarity': float(non_diag_sims.min()),
        'max_similarity': float(non_diag_sims.max()),
        'median_similarity': float(np.median(non_diag_sims))
    }
    
    logger.info(f"\n📊 SimRank Statistics:")
    logger.info(f"   Total document pairs: {stats['total_pairs']:,}")
    logger.info(f"   Mean similarity: {stats['mean_similarity']:.4f}")
    logger.info(f"   Std deviation: {stats['std_similarity']:.4f}")
    logger.info(f"   Min similarity: {stats['min_similarity']:.4f}")
    logger.info(f"   Max similarity: {stats['max_similarity']:.4f}")
    logger.info(f"   Median similarity: {stats['median_similarity']:.4f}")
    
    # Distribution analysis
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    logger.info(f"\n📈 Similarity Distribution:")
    
    for thresh in thresholds:
        count = np.sum(non_diag_sims >= thresh)
        percentage = (count / len(non_diag_sims)) * 100
        logger.info(f"   >= {thresh:.1f}: {count:,} pairs ({percentage:.2f}%)")
    
    # Find most similar document pairs
    logger.info(f"\n🔝 Top 10 Most Similar Document Pairs:")
    
    # Get top pairs (excluding diagonal)
    top_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > 0.01:  # Only non-trivial similarities
                top_pairs.append((i, j, sim_matrix[i, j]))
    
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for rank, (i, j, sim) in enumerate(top_pairs[:10], 1):
        doc_i = G.nodes[doc_nodes[i]]
        doc_j = G.nodes[doc_nodes[j]]
        
        logger.info(f"\n   {rank}. Similarity: {sim:.4f}")
        logger.info(f"      A: {doc_i['title'][:60]}")
        logger.info(f"      B: {doc_j['title'][:60]}")
        logger.info(f"      Cluster: {doc_i['cluster_label']} <-> {doc_j['cluster_label']}")
    
    return stats, top_pairs


def save_simrank_results(sim_matrix, node_to_idx, doc_nodes, G, stats):
    """Save SimRank results"""
    logger.info("\n[Phase 4] Saving results...")
    
    os.makedirs(SIMRANK_DIR, exist_ok=True)
    
    # Save similarity matrix
    np.save(SIMRANK_MATRIX_FILE, sim_matrix)
    logger.info(f"✓ Saved similarity matrix: {SIMRANK_MATRIX_FILE}")
    
    # Create mapping file
    mapping = {
        'computation_date': datetime.now().isoformat(),
        'parameters': {
            'max_iterations': MAX_ITERATIONS,
            'decay_factor': DECAY_FACTOR,
            'convergence_threshold': CONVERGENCE_THRESHOLD
        },
        'statistics': stats,
        'documents': []
    }
    
    # Add document information
    for node in doc_nodes:
        idx = node_to_idx[node]
        doc_data = G.nodes[node]
        
        doc_info = {
            'index': idx,
            'node_id': node,
            'title': doc_data['title'],
            'cluster_id': doc_data['cluster_id'],
            'cluster_label': doc_data['cluster_label'],
            'heritage_types': doc_data['heritage_types'],
            'domains': doc_data['domains'],
            'time_period': doc_data['time_period'],
            'region': doc_data['region']
        }
        
        mapping['documents'].append(doc_info)
    
    with open(SIMRANK_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved mapping file: {SIMRANK_MAPPING_FILE}")


def test_simrank(sim_matrix, node_to_idx, doc_nodes, G):
    """Test SimRank with sample queries"""
    logger.info("\n[Phase 5] Testing SimRank recommendations...")
    
    # Test with first document
    test_idx = 0
    test_node = doc_nodes[test_idx]
    test_doc = G.nodes[test_node]
    
    logger.info(f"\n🧪 Test Query:")
    logger.info(f"   Document: {test_doc['title']}")
    logger.info(f"   Type: {test_doc['heritage_types']}")
    logger.info(f"   Cluster: {test_doc['cluster_label']}")
    
    # Get top-k similar documents
    similarities = sim_matrix[test_idx]
    top_k_indices = np.argsort(similarities)[::-1][1:11]  # Top 10, skip itself
    
    logger.info(f"\n   Top 10 Similar Documents (by SimRank):")
    
    for rank, idx in enumerate(top_k_indices, 1):
        similar_node = doc_nodes[idx]
        similar_doc = G.nodes[similar_node]
        sim_score = similarities[idx]
        
        logger.info(f"\n   {rank}. {similar_doc['title'][:60]}")
        logger.info(f"      SimRank: {sim_score:.4f}")
        logger.info(f"      Type: {similar_doc['heritage_types']}")
        logger.info(f"      Cluster: {similar_doc['cluster_label']}")


def main():
    logger.info("="*70)
    logger.info("SIMRANK COMPUTATION")
    logger.info("="*70)
    
    # Load knowledge graph
    result = load_knowledge_graph()
    if result is None:
        return
    
    G, doc_nodes = result
    
    # Compute SimRank
    sim_matrix, node_to_idx = compute_simrank_optimized(G, doc_nodes)
    
    # Analyze results
    stats, top_pairs = analyze_simrank_results(sim_matrix, doc_nodes, G)
    
    # Save results
    save_simrank_results(sim_matrix, node_to_idx, doc_nodes, G, stats)
    
    # Test recommendations
    test_simrank(sim_matrix, node_to_idx, doc_nodes, G)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SIMRANK COMPUTATION COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Computed SimRank for {len(doc_nodes)} documents")
    logger.info(f"📊 Matrix size: {sim_matrix.shape}")
    logger.info(f"💾 Files created:")
    logger.info(f"   - {SIMRANK_MATRIX_FILE}")
    logger.info(f"   - {SIMRANK_MAPPING_FILE}")
    logger.info("="*70)


if __name__ == "__main__":
    main()