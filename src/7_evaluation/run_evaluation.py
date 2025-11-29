"""
Run Complete Evaluation
Execute all metrics and generate report
"""

import json
import os
import sys
import numpy as np
import pickle
import faiss
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config_loader import get_config
from utils.logger import get_logger

# Import metrics from same directory
import importlib.util
spec = importlib.util.spec_from_file_location("metrics", os.path.join(os.path.dirname(__file__), "metrics.py"))
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)

compute_all_metrics = metrics_module.compute_all_metrics
PerformanceTimer = metrics_module.PerformanceTimer

config = get_config()
logger = get_logger(__name__)

def load_all_data():
    """Load everything needed for evaluation"""
    logger.info("Loading all data...")
    
    # Documents
    classified_file = config.get_path('data', 'classified') + '/classified_documents.json'
    with open(classified_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Embeddings
    embeddings_file = config.get_path('data', 'embeddings') + '/document_embeddings.npy'
    embeddings = np.load(embeddings_file)
    
    # FAISS index
    faiss_file = config.get_path('models', 'ranker') + '/faiss/flat_index.faiss'
    faiss_index = faiss.read_index(faiss_file)
    
    # SimRank
    simrank_file = 'knowledge_graph/simrank/simrank_matrix.npy'
    simrank_matrix = np.load(simrank_file)
    
    # Ground truth
    gt_file = config.get_path('data', 'evaluation') + '/ground_truth.json'
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth_queries = json.load(f)
    
    logger.info(f"✓ Loaded {len(documents)} documents")
    logger.info(f"✓ Loaded FAISS index with {faiss_index.ntotal} vectors")
    logger.info(f"✓ Loaded {len(ground_truth_queries)} test queries")
    
    return documents, embeddings, faiss_index, simrank_matrix, ground_truth_queries

def run_recommendations(query_idx, embeddings, faiss_index, simrank_matrix, k=20, hybrid_weight=0.5):
    """
    Generate recommendations using hybrid approach
    
    Args:
        query_idx: Query document index
        embeddings: Document embeddings
        faiss_index: FAISS index
        simrank_matrix: SimRank similarity matrix
        k: Number of recommendations
        hybrid_weight: Weight for embedding vs SimRank (0.5 = equal weight)
    
    Returns:
        List of recommended document indices
    """
    # Get query embedding
    query_emb = embeddings[query_idx:query_idx+1].astype('float32')
    
    # FAISS search (embedding-based)
    distances_emb, indices_emb = faiss_index.search(query_emb, k*2)
    
    # Convert FAISS distances to similarities (L2 distance to similarity)
    similarities_emb = 1 / (1 + distances_emb[0])
    
    # SimRank scores
    simrank_scores = simrank_matrix[query_idx]
    
    # Hybrid scoring
    hybrid_scores = {}
    
    for idx, sim_emb in zip(indices_emb[0], similarities_emb):
        if idx == query_idx:
            continue
        
        sim_sr = simrank_scores[idx]
        
        # Normalize and combine
        hybrid_score = hybrid_weight * sim_emb + (1 - hybrid_weight) * (sim_sr * 10)  # Scale SimRank
        hybrid_scores[idx] = hybrid_score
    
    # Sort by hybrid score
    ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [idx for idx, score in ranked[:k]]

def evaluate_all_methods(documents, embeddings, faiss_index, simrank_matrix, ground_truth_queries):
    """
    Evaluate different recommendation methods
    """
    logger.info("\n" + "="*70)
    logger.info("EVALUATING RECOMMENDATION METHODS")
    logger.info("="*70)
    
    methods = {
        'embedding_only': {'weight': 1.0, 'name': 'Embedding-Only (FAISS)'},
        'simrank_only': {'weight': 0.0, 'name': 'SimRank-Only'},
        'hybrid_balanced': {'weight': 0.5, 'name': 'Hybrid (50-50)'},
        'hybrid_emb_heavy': {'weight': 0.7, 'name': 'Hybrid (70% Embedding)'},
        'hybrid_sr_heavy': {'weight': 0.3, 'name': 'Hybrid (70% SimRank)'}
    }
    
    results = {}
    
    for method_id, method_config in methods.items():
        logger.info(f"\n[Evaluating] {method_config['name']}...")
        
        query_results = []
        
        with PerformanceTimer() as timer:
            for query in ground_truth_queries[:50]:  # Evaluate on first 50 queries
                query_idx = query['query_idx']
                
                # Generate recommendations
                recommended = run_recommendations(
                    query_idx, 
                    embeddings, 
                    faiss_index, 
                    simrank_matrix, 
                    k=20,
                    hybrid_weight=method_config['weight']
                )
                
                query_results.append({
                    'query_id': query['query_id'],
                    'query_item': query_idx,
                    'recommended': recommended,
                    'explanations': []  # Will add later
                })
        
        # Compute metrics
        ground_truth_dict = {q['query_id']: set(q['relevant_docs']) for q in ground_truth_queries[:50]}
        
        # Create metadata and embeddings dicts
        item_metadata = {i: doc for i, doc in enumerate(documents)}
        item_embeddings = {i: emb for i, emb in enumerate(embeddings)}
        
        metrics = compute_all_metrics(
            query_results,
            ground_truth_dict,
            item_metadata,
            item_embeddings,
            k_values=[5, 10, 20]
        )
        
        # Add timing
        metrics['efficiency'] = {
            'total_time_seconds': timer.elapsed,
            'avg_query_latency_ms': (timer.elapsed / len(query_results)) * 1000,
            'queries_per_second': len(query_results) / timer.elapsed
        }
        
        results[method_id] = {
            'name': method_config['name'],
            'config': method_config,
            'metrics': metrics
        }
        
        logger.info(f"  ✓ Precision@10: {metrics['accuracy'].get('precision@10', 0):.4f}")
        logger.info(f"  ✓ NDCG@10: {metrics['accuracy'].get('ndcg@10', 0):.4f}")
        logger.info(f"  ✓ Query latency: {metrics['efficiency']['avg_query_latency_ms']:.2f}ms")
    
    return results

def generate_report(results, output_dir):
    """Generate evaluation report"""
    logger.info("\n[Generating] Evaluation report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Full results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"  ✓ Full results: {results_file}")
    
    # Comparison table
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'methods': {}
    }
    
    for method_id, result in results.items():
        metrics = result['metrics']
        comparison['methods'][result['name']] = {
            'precision@5': metrics['accuracy'].get('precision@5', 0),
            'precision@10': metrics['accuracy'].get('precision@10', 0),
            'recall@10': metrics['accuracy'].get('recall@10', 0),
            'ndcg@10': metrics['accuracy'].get('ndcg@10', 0),
            'MAP': metrics['accuracy'].get('MAP', 0),
            'diversity': metrics['diversity'].get('intra_list_diversity', 0),
            'coverage': metrics['diversity'].get('coverage', 0),
            'temporal_accuracy': metrics['heritage_specific'].get('temporal_accuracy', 0),
            'query_latency_ms': metrics['efficiency']['avg_query_latency_ms']
        }
    
    comparison_file = os.path.join(output_dir, 'method_comparison.json')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"  ✓ Comparison: {comparison_file}")
    
    return comparison

def print_comparison_table(comparison):
    """Print formatted comparison table"""
    logger.info("\n" + "="*70)
    logger.info("METHOD COMPARISON")
    logger.info("="*70)
    
    methods = list(comparison['methods'].keys())
    metrics = ['precision@10', 'ndcg@10', 'MAP', 'diversity', 'temporal_accuracy', 'query_latency_ms']
    
    # Header
    header = f"{'Method':<30} | " + " | ".join([f"{m:>12}" for m in metrics])
    logger.info(header)
    logger.info("-" * len(header))
    
    # Rows
    for method in methods:
        values = comparison['methods'][method]
        row = f"{method:<30} | "
        row += " | ".join([f"{values.get(m, 0):>12.4f}" for m in metrics])
        logger.info(row)
    
    logger.info("="*70)

def main():
    logger.info("="*70)
    logger.info("COMPREHENSIVE EVALUATION")
    logger.info("="*70)
    
    # Load everything
    documents, embeddings, faiss_index, simrank_matrix, ground_truth_queries = load_all_data()
    
    # Run evaluation
    results = evaluate_all_methods(documents, embeddings, faiss_index, simrank_matrix, ground_truth_queries)
    
    # Generate report
    output_dir = config.get_path('results', 'base')
    comparison = generate_report(results, output_dir)
    
    # Print table
    print_comparison_table(comparison)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Evaluated {len(results)} methods")
    logger.info(f"✅ Used {len(ground_truth_queries[:50])} test queries")
    logger.info(f"\n📂 Results saved to: {output_dir}")
    logger.info("="*70)

if __name__ == "__main__":
    main()