"""
Unified Evaluation Pipeline
Evaluates recommendation system using improved ground truth
"""

import json
import os
import numpy as np
import faiss
from datetime import datetime
from collections import defaultdict

# Import metrics
import sys
sys.path.append('src/7_evaluation')
from metrics import (
    precision_at_k, recall_at_k, f1_at_k, ndcg_at_k,
    mean_average_precision, intra_list_diversity,
    catalog_coverage, temporal_accuracy, spatial_relevance,
    cultural_domain_alignment, PerformanceTimer
)

def load_all_data():
    """Load everything needed for evaluation."""
    print("📂 Loading system components...")
    
    # Documents
    with open('data/classified/classified_documents.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Embeddings
    embeddings = np.load('data/embeddings/document_embeddings.npy')
    
    # FAISS index
    faiss_index = faiss.read_index('models/ranker/faiss/flat_index.faiss')
    
    # SimRank
    simrank_matrix = np.load('knowledge_graph/simrank/simrank_matrix.npy')
    
    # Ground truth (try improved first, fallback to old)
    gt_improved = 'data/evaluation/ground_truth_improved.json'
    gt_old = 'data/evaluation/ground_truth.json'
    
    if os.path.exists(gt_improved):
        print("   Using improved ground truth ✨")
        with open(gt_improved, 'r', encoding='utf-8') as f:
            ground_truth_queries = json.load(f)
    elif os.path.exists(gt_old):
        print("   ⚠️ Using old ground truth (consider regenerating)")
        with open(gt_old, 'r', encoding='utf-8') as f:
            ground_truth_queries = json.load(f)
    else:
        print("   ❌ No ground truth found! Run: python src/7_evaluation/create_ground_truth.py")
        return None
    
    print(f"✅ Loaded: {len(documents)} docs, {len(ground_truth_queries)} test queries")
    
    return documents, embeddings, faiss_index, simrank_matrix, ground_truth_queries


def run_recommendations(query_idx, embeddings, faiss_index, simrank_matrix, 
                       k=20, hybrid_weight=0.5):
    """
    Generate recommendations using hybrid approach.
    
    Args:
        query_idx: Query document index
        embeddings: Document embeddings
        faiss_index: FAISS index
        simrank_matrix: SimRank similarity matrix
        k: Number of recommendations
        hybrid_weight: Weight for embedding vs SimRank
            0.0 = SimRank only
            1.0 = Embedding only
            0.5 = Balanced hybrid
    
    Returns:
        List of recommended document indices
    """
    # Get query embedding
    query_emb = embeddings[query_idx:query_idx+1].astype('float32')
    
    # FAISS search (embedding-based)
    distances_emb, indices_emb = faiss_index.search(query_emb, k*2)
    
    # Convert distances to similarities
    # For L2 distance: similarity = 1 / (1 + distance)
    similarities_emb = 1 / (1 + distances_emb[0])
    
    # SimRank scores
    simrank_scores = simrank_matrix[query_idx]
    
    # Hybrid scoring
    hybrid_scores = {}
    
    for idx, sim_emb in zip(indices_emb[0], similarities_emb):
        if idx == query_idx:
            continue
        
        sim_sr = simrank_scores[idx]
        
        # Normalize SimRank (typically 0-0.2 range) to 0-1
        sim_sr_norm = min(1.0, sim_sr * 5.0)
        
        # Weighted combination
        hybrid_score = hybrid_weight * sim_emb + (1 - hybrid_weight) * sim_sr_norm
        hybrid_scores[idx] = hybrid_score
    
    # Sort by hybrid score
    ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [idx for idx, score in ranked[:k]]


def evaluate_method(method_name, hybrid_weight, documents, embeddings, 
                   faiss_index, simrank_matrix, ground_truth_queries):
    """Evaluate a recommendation method."""
    print(f"\n🔬 Evaluating: {method_name}")
    print(f"   Hybrid weight: {hybrid_weight:.2f} (0=SimRank, 1=Embedding)")
    
    query_results = []
    
    with PerformanceTimer() as timer:
        for query in ground_truth_queries[:50]:  # Use first 50 for speed
            query_idx = query['query_idx']
            
            # Generate recommendations
            recommended = run_recommendations(
                query_idx, embeddings, faiss_index, simrank_matrix,
                k=20, hybrid_weight=hybrid_weight
            )
            
            query_results.append({
                'query_id': query['query_id'],
                'query_item': query_idx,
                'recommended': recommended
            })
    
    # Compute metrics
    print("   Computing metrics...")
    
    # Prepare ground truth dict
    ground_truth_dict = {q['query_id']: set(q['relevant_docs']) for q in ground_truth_queries[:50]}
    
    # Accuracy metrics
    accuracy = {}
    
    for k in [5, 10, 20]:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ndcg_scores = []
        
        for result in query_results:
            relevant = ground_truth_dict[result['query_id']]
            recommended = result['recommended']
            
            precision_scores.append(precision_at_k(relevant, recommended, k))
            recall_scores.append(recall_at_k(relevant, recommended, k))
            f1_scores.append(f1_at_k(relevant, recommended, k))
            ndcg_scores.append(ndcg_at_k(relevant, recommended, k))
        
        accuracy[f'precision@{k}'] = np.mean(precision_scores)
        accuracy[f'recall@{k}'] = np.mean(recall_scores)
        accuracy[f'f1@{k}'] = np.mean(f1_scores)
        accuracy[f'ndcg@{k}'] = np.mean(ndcg_scores)
    
    # MAP
    relevant_per_query = [ground_truth_dict[r['query_id']] for r in query_results]
    recommended_per_query = [r['recommended'] for r in query_results]
    accuracy['MAP'] = mean_average_precision(relevant_per_query, recommended_per_query)
    
    # Diversity
    item_embeddings = {i: emb for i, emb in enumerate(embeddings)}
    
    diversity = {
        'intra_list_diversity': np.mean([
            intra_list_diversity(r['recommended'][:20], item_embeddings)
            for r in query_results
        ]),
        'coverage': catalog_coverage([r['recommended'] for r in query_results], len(documents))
    }
    
    # Heritage-specific metrics
    item_metadata = {i: doc for i, doc in enumerate(documents)}
    
    heritage_specific = {}
    
    temporal_scores = []
    spatial_scores = []
    cultural_scores = []
    
    for result in query_results:
        query_idx = result['query_item']
        recommended = result['recommended'][:10]
        
        query_doc = documents[query_idx]
        query_classes = query_doc['classifications']
        
        # Temporal
        if query_classes.get('time_period'):
            temp_acc = temporal_accuracy(recommended, query_classes['time_period'], item_metadata)
            temporal_scores.append(temp_acc)
        
        # Spatial
        if query_classes.get('region'):
            spat_rel = spatial_relevance(recommended, query_classes['region'], item_metadata)
            spatial_scores.append(spat_rel)
        
        # Cultural domain
        query_domains = set(query_classes.get('domains', []))
        if query_domains:
            cult_align = cultural_domain_alignment(recommended, query_domains, item_metadata)
            cultural_scores.append(cult_align)
    
    if temporal_scores:
        heritage_specific['temporal_accuracy'] = np.mean(temporal_scores)
    if spatial_scores:
        heritage_specific['spatial_relevance'] = np.mean(spatial_scores)
    if cultural_scores:
        heritage_specific['cultural_domain_alignment'] = np.mean(cultural_scores)
    
    # Efficiency
    efficiency = {
        'total_time_seconds': timer.elapsed,
        'avg_query_latency_ms': (timer.elapsed / len(query_results)) * 1000,
        'queries_per_second': len(query_results) / timer.elapsed
    }
    
    return {
        'accuracy': accuracy,
        'diversity': diversity,
        'heritage_specific': heritage_specific,
        'efficiency': efficiency
    }


def main():
    print("="*80)
    print("UNIFIED EVALUATION PIPELINE")
    print("="*80)
    
    # Load data
    data = load_all_data()
    if data is None:
        print("\n❌ Cannot proceed without ground truth")
        return
    
    documents, embeddings, faiss_index, simrank_matrix, ground_truth_queries = data
    
    # Define methods to evaluate
    methods = {
        'embedding_only': {
            'name': 'Embedding-Only (FAISS)',
            'weight': 1.0
        },
        'simrank_only': {
            'name': 'SimRank-Only',
            'weight': 0.0
        },
        'hybrid_balanced': {
            'name': 'Hybrid (50-50)',
            'weight': 0.5
        },
        'hybrid_emb_heavy': {
            'name': 'Hybrid (70% Embedding)',
            'weight': 0.7
        },
        'hybrid_sr_heavy': {
            'name': 'Hybrid (30% Embedding)',
            'weight': 0.3
        }
    }
    
    # Evaluate all methods
    results = {}
    
    for method_id, config in methods.items():
        metrics = evaluate_method(
            config['name'],
            config['weight'],
            documents, embeddings, faiss_index,
            simrank_matrix, ground_truth_queries
        )
        
        results[method_id] = {
            'name': config['name'],
            'config': config,
            'metrics': metrics
        }
        
        # Print summary
        print(f"   ✅ Precision@10: {metrics['accuracy']['precision@10']:.4f}")
        print(f"   ✅ NDCG@10: {metrics['accuracy']['ndcg@10']:.4f}")
        print(f"   ✅ MAP: {metrics['accuracy']['MAP']:.4f}")
    
    # Save results
    print("\n💾 Saving results...")
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Saved to: {results_file}")
    
    # Create comparison table
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'methods': {}
    }
    
    for method_id, result in results.items():
        metrics = result['metrics']
        comparison['methods'][result['name']] = {
            'precision@5': metrics['accuracy']['precision@5'],
            'precision@10': metrics['accuracy']['precision@10'],
            'recall@10': metrics['accuracy']['recall@10'],
            'ndcg@10': metrics['accuracy']['ndcg@10'],
            'MAP': metrics['accuracy']['MAP'],
            'diversity': metrics['diversity']['intra_list_diversity'],
            'coverage': metrics['diversity']['coverage'],
            'temporal_accuracy': metrics['heritage_specific'].get('temporal_accuracy', 0),
            'query_latency_ms': metrics['efficiency']['avg_query_latency_ms']
        }
    
    comparison_file = os.path.join(output_dir, 'method_comparison.json')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"   ✅ Comparison saved to: {comparison_file}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n{'Method':<30} {'P@10':<8} {'NDCG@10':<8} {'MAP':<8} {'Latency(ms)':<12}")
    print("-"*80)
    
    for method_name, metrics in comparison['methods'].items():
        print(f"{method_name:<30} {metrics['precision@10']:<8.4f} {metrics['ndcg@10']:<8.4f} "
              f"{metrics['MAP']:<8.4f} {metrics['query_latency_ms']:<12.2f}")
    
    # Best method
    best_method = max(comparison['methods'].items(), 
                     key=lambda x: x[1]['precision@10'])
    
    print("\n" + "="*80)
    print(f"🏆 BEST METHOD: {best_method[0]}")
    print(f"   Precision@10: {best_method[1]['precision@10']:.4f}")
    print(f"   NDCG@10: {best_method[1]['ndcg@10']:.4f}")
    print(f"   MAP: {best_method[1]['MAP']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()