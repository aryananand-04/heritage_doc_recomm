"""
Comprehensive Evaluation Metrics for Heritage Recommendation System
Includes: Accuracy, Diversity, Explainability, Heritage-Specific Metrics
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# ========== ACCURACY METRICS ==========

def precision_at_k(relevant_items, recommended_items, k):
    """Precision@K: Proportion of recommended items that are relevant"""
    if k == 0 or len(recommended_items) == 0:
        return 0.0
    
    recommended_at_k = set(recommended_items[:k])
    hits = len(recommended_at_k & relevant_items)
    
    return hits / k

def recall_at_k(relevant_items, recommended_items, k):
    """Recall@K: Proportion of relevant items that were recommended"""
    if len(relevant_items) == 0:
        return 0.0
    
    recommended_at_k = set(recommended_items[:k])
    hits = len(recommended_at_k & relevant_items)
    
    return hits / len(relevant_items)

def f1_at_k(relevant_items, recommended_items, k):
    """F1@K: Harmonic mean of Precision@K and Recall@K"""
    p = precision_at_k(relevant_items, recommended_items, k)
    r = recall_at_k(relevant_items, recommended_items, k)
    
    if p + r == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)

def ndcg_at_k(relevant_items, recommended_items, k, relevance_scores=None):
    """NDCG@K: Normalized Discounted Cumulative Gain"""
    if k == 0:
        return 0.0
    
    # DCG calculation
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
            dcg += rel / np.log2(i + 2)
    
    # IDCG calculation
    if relevance_scores:
        ideal_rels = sorted([relevance_scores.get(item, 0) for item in relevant_items], reverse=True)
    else:
        ideal_rels = [1.0] * len(relevant_items)
    
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def mean_average_precision(relevant_items_per_query, recommended_items_per_query):
    """MAP: Mean Average Precision across all queries"""
    aps = []
    
    for relevant, recommended in zip(relevant_items_per_query, recommended_items_per_query):
        if len(relevant) == 0:
            continue
        
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(recommended):
            if item in relevant:
                hits += 1
                precision = hits / (i + 1)
                sum_precisions += precision
        
        ap = sum_precisions / len(relevant) if len(relevant) > 0 else 0.0
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0

# ========== DIVERSITY METRICS ==========

def intra_list_diversity(recommended_items, item_embeddings):
    """Intra-List Diversity: Average pairwise distance between recommended items"""
    if len(recommended_items) < 2:
        return 0.0
    
    embeddings = np.array([item_embeddings[item] for item in recommended_items if item in item_embeddings])
    
    if len(embeddings) < 2:
        return 0.0
    
    similarity_matrix = cosine_similarity(embeddings)
    
    n = len(embeddings)
    total_distance = 0.0
    count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            total_distance += (1 - similarity_matrix[i][j])
            count += 1
    
    return total_distance / count if count > 0 else 0.0

def catalog_coverage(recommended_items_all_queries, total_catalog_size):
    """Coverage: Percentage of catalog items that appear in recommendations"""
    unique_recommended = set()
    for recs in recommended_items_all_queries:
        unique_recommended.update(recs)
    
    return len(unique_recommended) / total_catalog_size if total_catalog_size > 0 else 0.0

# ========== HERITAGE-SPECIFIC METRICS ==========

def temporal_accuracy(recommended_items, query_time_period, item_metadata):
    """Temporal Accuracy: % of recommendations from the same time period"""
    if not recommended_items:
        return 0.0
    
    same_period = 0
    for item in recommended_items:
        meta = item_metadata.get(item, {})
        item_period = meta.get('classifications', {}).get('time_period', 'unknown')
        if item_period == query_time_period:
            same_period += 1
    
    return same_period / len(recommended_items)

def spatial_relevance(recommended_items, query_region, item_metadata):
    """Spatial Relevance: % of recommendations from the same region"""
    if not recommended_items:
        return 0.0
    
    same_region = 0
    for item in recommended_items:
        meta = item_metadata.get(item, {})
        item_region = meta.get('classifications', {}).get('region', 'unknown')
        if item_region == query_region:
            same_region += 1
    
    return same_region / len(recommended_items)

def cultural_domain_alignment(recommended_items, query_domains, item_metadata):
    """Cultural Domain Alignment: Overlap of domains between query and recommendations"""
    if not recommended_items or not query_domains:
        return 0.0
    
    domain_overlaps = []
    
    for item in recommended_items:
        meta = item_metadata.get(item, {})
        item_domains = set(meta.get('classifications', {}).get('domains', []))
        
        if item_domains:
            overlap = len(query_domains & item_domains) / len(query_domains | item_domains)
            domain_overlaps.append(overlap)
    
    return np.mean(domain_overlaps) if domain_overlaps else 0.0

# ========== PERFORMANCE TIMER ==========

class PerformanceTimer:
    """Context manager for timing operations"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.time()
    
    @property
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

# ========== COMPUTE ALL METRICS ==========

def compute_all_metrics(query_results, ground_truth, item_metadata, item_embeddings, k_values=[5, 10, 20]):
    """Compute all metrics for a set of queries"""
    results = {
        'accuracy': defaultdict(list),
        'diversity': {},
        'heritage_specific': defaultdict(list),
        'explainability': {}
    }
    
    for query in query_results:
        query_id = query['query_id']
        recommended = query['recommended']
        relevant = ground_truth.get(query_id, set())
        
        # Accuracy metrics
        for k in k_values:
            results['accuracy'][f'precision@{k}'].append(
                precision_at_k(relevant, recommended, k)
            )
            results['accuracy'][f'recall@{k}'].append(
                recall_at_k(relevant, recommended, k)
            )
            results['accuracy'][f'f1@{k}'].append(
                f1_at_k(relevant, recommended, k)
            )
            results['accuracy'][f'ndcg@{k}'].append(
                ndcg_at_k(relevant, recommended, k)
            )
        
        # Diversity
        if item_embeddings:
            diversity = intra_list_diversity(recommended[:20], item_embeddings)
            results['diversity'].setdefault('intra_list_diversity', []).append(diversity)
        
        # Heritage-specific
        query_item = query.get('query_item')
        if query_item and query_item in item_metadata:
            query_meta = item_metadata[query_item]
            query_classes = query_meta.get('classifications', {})
            
            query_period = query_classes.get('time_period')
            if query_period:
                temp_acc = temporal_accuracy(recommended[:10], query_period, item_metadata)
                results['heritage_specific']['temporal_accuracy'].append(temp_acc)
            
            query_region = query_classes.get('region')
            if query_region:
                spat_rel = spatial_relevance(recommended[:10], query_region, item_metadata)
                results['heritage_specific']['spatial_relevance'].append(spat_rel)
            
            query_domains = set(query_classes.get('domains', []))
            if query_domains:
                cult_align = cultural_domain_alignment(recommended[:10], query_domains, item_metadata)
                results['heritage_specific']['cultural_domain_alignment'].append(cult_align)
    
    # Average everything
    final_results = {
        'accuracy': {k: np.mean(v) for k, v in results['accuracy'].items()},
        'diversity': {k: np.mean(v) for k, v in results['diversity'].items()},
        'heritage_specific': {k: np.mean(v) for k, v in results['heritage_specific'].items()},
        'explainability': {}
    }
    
    # Add MAP
    relevant_per_query = [ground_truth.get(q['query_id'], set()) for q in query_results]
    recommended_per_query = [q['recommended'] for q in query_results]
    final_results['accuracy']['MAP'] = mean_average_precision(relevant_per_query, recommended_per_query)
    
    # Add Coverage
    all_recommended = [q['recommended'] for q in query_results]
    total_catalog = len(item_metadata)
    final_results['diversity']['coverage'] = catalog_coverage(all_recommended, total_catalog)
    
    return final_results