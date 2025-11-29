"""
Ground Truth Analyzer
Inspect and validate generated test queries
"""

import json
import numpy as np
from collections import Counter

def analyze_ground_truth(filepath='data/evaluation/ground_truth_improved.json'):
    """Analyze ground truth quality."""
    
    print("="*80)
    print("GROUND TRUTH ANALYSIS")
    print("="*80)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            queries = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"\n📊 OVERVIEW")
    print(f"   Total queries: {len(queries)}")
    
    # Relevant docs distribution
    relevant_counts = [len(q['relevant_docs']) for q in queries]
    
    print(f"\n📈 RELEVANT DOCUMENTS PER QUERY:")
    print(f"   Mean: {np.mean(relevant_counts):.1f}")
    print(f"   Median: {np.median(relevant_counts):.1f}")
    print(f"   Min: {np.min(relevant_counts)}")
    print(f"   Max: {np.max(relevant_counts)}")
    print(f"   Std: {np.std(relevant_counts):.1f}")
    
    # Histogram
    print(f"\n   Distribution:")
    bins = [0, 3, 5, 10, 20, 50, 1000]
    labels = ['1-2', '3-4', '5-9', '10-19', '20-49', '50+']
    
    for i in range(len(bins)-1):
        count = sum(1 for c in relevant_counts if bins[i] < c <= bins[i+1])
        if count > 0:
            print(f"      {labels[i]:8}: {count:3d} queries ({count/len(queries)*100:5.1f}%)")
    
    # Sample queries
    print(f"\n💡 SAMPLE QUERIES:")
    
    for i, query in enumerate(queries[:5], 1):
        print(f"\n   {i}. {query['query_doc']}")
        print(f"      Query ID: {query['query_id']}")
        print(f"      Relevant docs: {len(query['relevant_docs'])}")
        
        # Show relevance details if available
        if 'relevance_details' in query and query['relevance_details']:
            print(f"      Top relevance scores:")
            for j, detail in enumerate(query['relevance_details'][:3], 1):
                print(f"         {j}. Score: {detail['relevance_score']:.3f} "
                      f"(emb:{detail['components']['embedding']:.2f}, "
                      f"meta:{detail['components']['metadata']:.2f}, "
                      f"sr:{detail['components']['simrank']:.2f})")
    
    # Quality checks
    print(f"\n🔍 QUALITY CHECKS:")
    
    # Check for queries with too few relevant docs
    few_relevant = sum(1 for c in relevant_counts if c < 3)
    if few_relevant > 0:
        print(f"   ⚠️ {few_relevant} queries have <3 relevant docs")
    else:
        print(f"   ✅ All queries have ≥3 relevant docs")
    
    # Check for queries with many relevant docs (too easy)
    many_relevant = sum(1 for c in relevant_counts if c > 50)
    if many_relevant > 0:
        print(f"   ⚠️ {many_relevant} queries have >50 relevant docs (may be too easy)")
    else:
        print(f"   ✅ No overly easy queries")
    
    # Check for duplicate queries
    query_ids = [q['query_id'] for q in queries]
    duplicates = [k for k, v in Counter(query_ids).items() if v > 1]
    if duplicates:
        print(f"   ⚠️ {len(duplicates)} duplicate query IDs found")
    else:
        print(f"   ✅ No duplicate query IDs")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    avg_relevant = np.mean(relevant_counts)
    
    if len(queries) < 30:
        print(f"   📌 Generate more queries (current: {len(queries)}, target: 50+)")
        print(f"      → Lower relevance threshold or minimum relevant docs")
    
    if avg_relevant < 5:
        print(f"   📌 Average relevant docs is low ({avg_relevant:.1f})")
        print(f"      → Consider lowering relevance threshold")
    elif avg_relevant > 20:
        print(f"   📌 Average relevant docs is high ({avg_relevant:.1f})")
        print(f"      → Consider raising relevance threshold")
    else:
        print(f"   ✅ Average relevant docs is good ({avg_relevant:.1f})")
    
    if len(queries) >= 30 and 5 <= avg_relevant <= 20:
        print(f"\n🎉 Ground truth quality looks good!")
        print(f"   Ready for evaluation: python evaluate_system.py")
    
    print("="*80)


if __name__ == "__main__":
    analyze_ground_truth()