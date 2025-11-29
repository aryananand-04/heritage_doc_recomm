"""
Ground Truth Generator
Create test queries and relevant document sets for evaluation
"""

import json
import os
import sys
import random
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__)

def load_data():
    """Load classified documents, embeddings, and KG"""
    logger.info("Loading data for ground truth generation...")
    
    # Load documents
    classified_file = config.get_path('data', 'classified') + '/classified_documents.json'
    with open(classified_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Load embeddings  
    embeddings_file = config.get_path('data', 'embeddings') + '/document_embeddings.npy'
    embeddings = np.load(embeddings_file)
    
    # Load SimRank scores
    simrank_file = 'knowledge_graph/simrank/simrank_matrix.npy'
    simrank_matrix = np.load(simrank_file)
    
    logger.info(f"✓ Loaded {len(documents)} documents")
    logger.info(f"✓ Loaded embeddings: {embeddings.shape}")
    logger.info(f"✓ Loaded SimRank matrix: {simrank_matrix.shape}")
    
    return documents, embeddings, simrank_matrix

def generate_cluster_based_ground_truth(documents, n_queries=50):
    """
    Generate ground truth based on cluster membership
    Documents in the same cluster are considered relevant
    """
    logger.info(f"\n[Strategy 1] Cluster-based ground truth...")
    
    # Group by cluster
    from collections import defaultdict
    clusters = defaultdict(list)
    
    for idx, doc in enumerate(documents):
        cluster_id = doc['cluster_id']
        clusters[cluster_id].append(idx)
    
    queries = []
    
    # Select query documents from each cluster
    for cluster_id, doc_indices in clusters.items():
        if len(doc_indices) < 5:  # Skip small clusters
            continue
        
        # Pick 3-5 queries per cluster
        n_from_cluster = min(5, max(3, len(doc_indices) // 10))
        query_docs = random.sample(doc_indices, n_from_cluster)
        
        for query_idx in query_docs:
            # All other docs in same cluster are relevant
            relevant = set(doc_indices) - {query_idx}
            
            queries.append({
                'query_id': f'cluster_{cluster_id}_{query_idx}',
                'query_idx': query_idx,
                'query_doc': documents[query_idx]['title'],
                'relevant_docs': list(relevant),
                'relevance_type': 'cluster',
                'cluster_id': cluster_id,
                'expected_size': len(relevant)
            })
    
    logger.info(f"  ✓ Generated {len(queries)} cluster-based queries")
    return queries

def generate_metadata_based_ground_truth(documents, n_queries=30):
    """
    Generate ground truth based on shared metadata
    (same heritage type, domain, time period, region)
    """
    logger.info(f"\n[Strategy 2] Metadata-based ground truth...")
    
    queries = []
    
    # Sample query documents
    query_indices = random.sample(range(len(documents)), min(n_queries, len(documents)))
    
    for query_idx in query_indices:
        query_doc = documents[query_idx]
        query_classes = query_doc['classifications']
        
        relevant = set()
        
        # Find docs with shared attributes
        for idx, doc in enumerate(documents):
            if idx == query_idx:
                continue
            
            doc_classes = doc['classifications']
            
            # Score based on attribute overlap
            score = 0
            
            # Heritage type overlap
            query_types = set(query_classes.get('heritage_types', []))
            doc_types = set(doc_classes.get('heritage_types', []))
            if query_types & doc_types:
                score += 2
            
            # Domain overlap
            query_domains = set(query_classes.get('domains', []))
            doc_domains = set(doc_classes.get('domains', []))
            if query_domains & doc_domains:
                score += 2
            
            # Same time period
            if query_classes.get('time_period') == doc_classes.get('time_period'):
                score += 1
            
            # Same region
            if query_classes.get('region') == doc_classes.get('region'):
                score += 1
            
            # Consider relevant if score >= 3
            if score >= 3:
                relevant.add(idx)
        
        if len(relevant) >= 5:  # Only keep if enough relevant docs
            queries.append({
                'query_id': f'metadata_{query_idx}',
                'query_idx': query_idx,
                'query_doc': query_doc['title'],
                'relevant_docs': list(relevant),
                'relevance_type': 'metadata',
                'query_attributes': {
                    'heritage_types': query_classes.get('heritage_types', []),
                    'domains': query_classes.get('domains', []),
                    'time_period': query_classes.get('time_period'),
                    'region': query_classes.get('region')
                },
                'expected_size': len(relevant)
            })
    
    logger.info(f"  ✓ Generated {len(queries)} metadata-based queries")
    return queries

def generate_embedding_based_ground_truth(documents, embeddings, n_queries=30, threshold=0.7):
    """
    Generate ground truth based on embedding similarity
    High cosine similarity = relevant
    """
    logger.info(f"\n[Strategy 3] Embedding-based ground truth (threshold={threshold})...")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    queries = []
    
    # Sample query documents
    query_indices = random.sample(range(len(documents)), min(n_queries, len(documents)))
    
    for query_idx in query_indices:
        # Find highly similar docs
        similarities = similarity_matrix[query_idx]
        relevant = set(np.where(similarities >= threshold)[0].tolist()) - {query_idx}
        
        if len(relevant) >= 5:
            queries.append({
                'query_id': f'embedding_{query_idx}',
                'query_idx': query_idx,
                'query_doc': documents[query_idx]['title'],
                'relevant_docs': list(relevant),
                'relevance_type': 'embedding',
                'threshold': threshold,
                'expected_size': len(relevant)
            })
    
    logger.info(f"  ✓ Generated {len(queries)} embedding-based queries")
    return queries

def generate_simrank_based_ground_truth(documents, simrank_matrix, n_queries=30, threshold=0.02):
    """
    Generate ground truth based on SimRank scores
    High SimRank = structurally similar in KG
    """
    logger.info(f"\n[Strategy 4] SimRank-based ground truth (threshold={threshold})...")
    
    queries = []
    
    # Sample query documents
    query_indices = random.sample(range(len(documents)), min(n_queries, len(documents)))
    
    for query_idx in query_indices:
        # Find docs with high SimRank
        simrank_scores = simrank_matrix[query_idx]
        relevant = set(np.where(simrank_scores >= threshold)[0].tolist()) - {query_idx}
        
        if len(relevant) >= 5:
            queries.append({
                'query_id': f'simrank_{query_idx}',
                'query_idx': query_idx,
                'query_doc': documents[query_idx]['title'],
                'relevant_docs': list(relevant),
                'relevance_type': 'simrank',
                'threshold': threshold,
                'expected_size': len(relevant)
            })
    
    logger.info(f"  ✓ Generated {len(queries)} SimRank-based queries")
    return queries

def save_ground_truth(queries, output_dir):
    """Save ground truth dataset"""
    logger.info(f"\n[Saving] Ground truth dataset...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all queries
    output_file = os.path.join(output_dir, 'ground_truth.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  ✓ Saved {len(queries)} queries to: {output_file}")
    
    # Save statistics
    from collections import Counter
    
    stats = {
        'total_queries': len(queries),
        'by_type': dict(Counter([q['relevance_type'] for q in queries])),
        'avg_relevant_docs': np.mean([q['expected_size'] for q in queries]),
        'min_relevant_docs': min([q['expected_size'] for q in queries]),
        'max_relevant_docs': max([q['expected_size'] for q in queries]),
        'query_types': list(set([q['relevance_type'] for q in queries]))
    }
    
    stats_file = os.path.join(output_dir, 'ground_truth_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"  ✓ Saved statistics to: {stats_file}")
    
    return stats

def main():
    logger.info("="*70)
    logger.info("GROUND TRUTH GENERATION")
    logger.info("="*70)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    documents, embeddings, simrank_matrix = load_data()
    
    # Generate ground truth using multiple strategies
    all_queries = []
    
    # Strategy 1: Cluster-based
    cluster_queries = generate_cluster_based_ground_truth(documents, n_queries=50)
    all_queries.extend(cluster_queries)
    
    # Strategy 2: Metadata-based
    metadata_queries = generate_metadata_based_ground_truth(documents, n_queries=30)
    all_queries.extend(metadata_queries)
    
    # Strategy 3: Embedding-based
    embedding_queries = generate_embedding_based_ground_truth(documents, embeddings, n_queries=30, threshold=0.7)
    all_queries.extend(embedding_queries)
    
    # Strategy 4: SimRank-based
    simrank_queries = generate_simrank_based_ground_truth(documents, simrank_matrix, n_queries=30, threshold=0.02)
    all_queries.extend(simrank_queries)
    
    # Save
    output_dir = config.get_path('data', 'evaluation')
    stats = save_ground_truth(all_queries, output_dir)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("GROUND TRUTH GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Total queries: {stats['total_queries']}")
    logger.info(f"✅ Query types: {stats['by_type']}")
    logger.info(f"✅ Avg relevant docs per query: {stats['avg_relevant_docs']:.1f}")
    logger.info(f"\n📂 Output: {output_dir}/ground_truth.json")
    logger.info("="*70)

if __name__ == "__main__":
    main()