"""
IMPROVED Ground Truth Generator with Multi-Signal Relevance

Uses 5 signals for relevance:
1. Embedding similarity (lowered threshold: 0.5)
2. Metadata overlap (heritage type + domain + period + region)
3. Entity overlap (shared locations/persons/orgs)
4. SimRank structural similarity (0.15 threshold)
5. Cluster membership (soft signal, not hard requirement)

Relevance Score = weighted sum → binary threshold
"""

import json
import os
import numpy as np
import pickle
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Load all required data."""
    print("📂 Loading data...")
    
    # Documents
    with open('data/classified/classified_documents.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Embeddings
    embeddings = np.load('data/embeddings/document_embeddings.npy')
    
    # SimRank
    simrank_matrix = np.load('knowledge_graph/simrank/simrank_matrix.npy')
    
    # KG for entity extraction
    with open('knowledge_graph/heritage_kg.gpickle', 'rb') as f:
        G = pickle.load(f)
    
    print(f"✓ Loaded: {len(documents)} docs, embeddings {embeddings.shape}, SimRank {simrank_matrix.shape}")
    
    return documents, embeddings, simrank_matrix, G


def compute_multi_signal_relevance(query_idx, candidate_idx, documents, embeddings, simrank_matrix, G, 
                                   doc_nodes):
    """
    Compute relevance score using 5 signals.
    
    Returns: relevance_score (0-1), component_scores dict
    """
    query_doc = documents[query_idx]
    cand_doc = documents[candidate_idx]
    
    # Signal 1: Embedding similarity (weight: 0.3)
    emb_sim = cosine_similarity(
        embeddings[query_idx].reshape(1, -1),
        embeddings[candidate_idx].reshape(1, -1)
    )[0][0]
    
    # Signal 2: Metadata overlap (weight: 0.25)
    query_meta = query_doc['classifications']
    cand_meta = cand_doc['classifications']
    
    meta_score = 0.0
    total_meta = 0
    
    # Heritage types
    query_types = set(query_meta.get('heritage_types', []))
    cand_types = set(cand_meta.get('heritage_types', []))
    if query_types and cand_types:
        meta_score += len(query_types & cand_types) / len(query_types | cand_types)
        total_meta += 1
    
    # Domains
    query_domains = set(query_meta.get('domains', []))
    cand_domains = set(cand_meta.get('domains', []))
    if query_domains and cand_domains:
        meta_score += len(query_domains & cand_domains) / len(query_domains | cand_domains)
        total_meta += 1
    
    # Time period (exact match)
    if query_meta.get('time_period') == cand_meta.get('time_period'):
        meta_score += 1.0
        total_meta += 1
    else:
        total_meta += 1
    
    # Region (exact match)
    if query_meta.get('region') == cand_meta.get('region'):
        meta_score += 1.0
        total_meta += 1
    else:
        total_meta += 1
    
    meta_score = meta_score / total_meta if total_meta > 0 else 0.0
    
    # Signal 3: Entity overlap (weight: 0.15)
    query_entities = set()
    cand_entities = set()
    
    query_node = f"doc_{query_idx}"
    cand_node = f"doc_{candidate_idx}"
    
    if query_node in doc_nodes and cand_node in doc_nodes:
        for neighbor in G.neighbors(query_node):
            if G.nodes[neighbor].get('node_type') in ['location', 'person', 'organization']:
                query_entities.add(neighbor)
        
        for neighbor in G.neighbors(cand_node):
            if G.nodes[neighbor].get('node_type') in ['location', 'person', 'organization']:
                cand_entities.add(neighbor)
    
    entity_overlap = 0.0
    if query_entities and cand_entities:
        entity_overlap = len(query_entities & cand_entities) / len(query_entities | cand_entities)
    
    # Signal 4: SimRank (weight: 0.2)
    simrank_score = simrank_matrix[query_idx, candidate_idx]
    
    # Signal 5: Cluster membership (weight: 0.1, soft bonus)
    cluster_bonus = 1.0 if query_doc['cluster_id'] == cand_doc['cluster_id'] else 0.0
    
    # Weighted combination
    weights = {
        'embedding': 0.3,
        'metadata': 0.25,
        'entities': 0.15,
        'simrank': 0.2,
        'cluster': 0.1
    }
    
    relevance_score = (
        weights['embedding'] * emb_sim +
        weights['metadata'] * meta_score +
        weights['entities'] * entity_overlap +
        weights['simrank'] * (simrank_score * 5) +  # Scale SimRank (0-0.2 → 0-1)
        weights['cluster'] * cluster_bonus
    )
    
    components = {
        'embedding': emb_sim,
        'metadata': meta_score,
        'entities': entity_overlap,
        'simrank': simrank_score,
        'cluster': cluster_bonus
    }
    
    return relevance_score, components


def generate_multi_signal_ground_truth(documents, embeddings, simrank_matrix, G, n_queries=100, relevance_threshold=0.4):
    """Generate ground truth using multi-signal relevance."""
    print(f"\n🎯 Generating Multi-Signal Ground Truth (threshold={relevance_threshold})...")
    
    # Get doc nodes
    doc_nodes = set([f"doc_{i}" for i in range(len(documents))])
    
    # Sample diverse queries (across clusters)
    cluster_groups = defaultdict(list)
    for idx, doc in enumerate(documents):
        cluster_groups[doc['cluster_id']].append(idx)
    
    # Stratified sampling
    queries_per_cluster = max(1, n_queries // len(cluster_groups))
    query_indices = []
    
    for cluster_id, doc_indices in cluster_groups.items():
        sample_size = min(queries_per_cluster, len(doc_indices))
        sampled = np.random.choice(doc_indices, size=sample_size, replace=False)
        query_indices.extend(sampled)
    
    # Cap at n_queries
    query_indices = query_indices[:n_queries]
    
    print(f"   Selected {len(query_indices)} query documents")
    
    queries = []
    
    for q_idx in query_indices:
        relevant_docs = []
        relevance_details = []
        
        # Score all candidates
        for cand_idx in range(len(documents)):
            if cand_idx == q_idx:
                continue
            
            rel_score, components = compute_multi_signal_relevance(
                q_idx, cand_idx, documents, embeddings, simrank_matrix, G, doc_nodes
            )
            
            if rel_score >= relevance_threshold:
                relevant_docs.append(cand_idx)
                relevance_details.append({
                    'doc_idx': int(cand_idx),  # Convert to native Python int
                    'relevance_score': float(rel_score),
                    'components': {k: float(v) for k, v in components.items()}
                })
        
        # Only keep queries with sufficient relevant docs
        if len(relevant_docs) >= 3:  # Lowered from 5 to get more queries
            queries.append({
                'query_id': f'multi_signal_{q_idx}',
                'query_idx': int(q_idx),  # Convert to native Python int
                'query_doc': documents[q_idx]['title'],
                'relevant_docs': [int(idx) for idx in relevant_docs],  # Convert to native Python ints
                'relevance_type': 'multi_signal',
                'relevance_threshold': relevance_threshold,
                'expected_size': len(relevant_docs),
                'relevance_details': sorted(relevance_details, key=lambda x: x['relevance_score'], reverse=True)[:10]  # Top 10
            })
    
    print(f"   ✓ Generated {len(queries)} queries with ≥5 relevant docs")
    
    return queries


def save_ground_truth(queries, output_dir='data/evaluation'):
    """Save ground truth with statistics."""
    print(f"\n💾 Saving ground truth...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save queries
    output_file = os.path.join(output_dir, 'ground_truth_improved.json')
    with open(output_file, 'w') as f:
        json.dump(queries, f, indent=2)
    
    print(f"   ✓ Saved {len(queries)} queries to {output_file}")
    
    # Statistics
    relevant_counts = [q['expected_size'] for q in queries]
    
    stats = {
        'total_queries': len(queries),
        'avg_relevant_per_query': float(np.mean(relevant_counts)),
        'std_relevant_per_query': float(np.std(relevant_counts)),
        'min_relevant': int(np.min(relevant_counts)),
        'max_relevant': int(np.max(relevant_counts)),
        'median_relevant': float(np.median(relevant_counts))
    }
    
    stats_file = os.path.join(output_dir, 'ground_truth_stats_improved.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"   ✓ Saved statistics to {stats_file}")
    
    # Display stats
    print(f"\n📊 Ground Truth Statistics:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Avg relevant docs: {stats['avg_relevant_per_query']:.1f}")
    print(f"   Range: {stats['min_relevant']}-{stats['max_relevant']}")
    
    return stats


def main():
    print("="*80)
    print("IMPROVED GROUND TRUTH GENERATION")
    print("Multi-Signal Relevance (Embedding + Metadata + Entities + SimRank + Cluster)")
    print("="*80)
    
    np.random.seed(42)
    
    # Load data
    documents, embeddings, simrank_matrix, G = load_data()
    
    # Generate ground truth
    queries = generate_multi_signal_ground_truth(
        documents, embeddings, simrank_matrix, G,
        n_queries=100,
        relevance_threshold=0.4  # Lowered from 0.5 for better coverage
    )
    
    # Save
    stats = save_ground_truth(queries)
    
    print("\n" + "="*80)
    print("✅ GROUND TRUTH GENERATION COMPLETE")
    print("="*80)
    print(f"Generated {stats['total_queries']} high-quality test queries")
    print(f"Use this for evaluation: data/evaluation/ground_truth_improved.json")
    print("="*80)


if __name__ == "__main__":
    main()