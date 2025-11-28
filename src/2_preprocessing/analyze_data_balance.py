import json
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories
CLASSIFIED_DIR = "data/classified"
OUTPUT_DIR = "data/balanced"

# Files
CLASSIFIED_FILE = os.path.join(CLASSIFIED_DIR, "classified_documents.json")


def load_classified_data():
    """Load classified documents"""
    logger.info(f"Loading classified data from: {CLASSIFIED_FILE}")
    
    if not os.path.exists(CLASSIFIED_FILE):
        logger.error(f"✗ File not found: {CLASSIFIED_FILE}")
        logger.error("Please run 4_train_autoencoder.py first to generate classified documents")
        return None
    
    with open(CLASSIFIED_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    logger.info(f"✓ Loaded {len(documents)} documents")
    
    return documents


def analyze_cluster_distribution(df):
    """Analyze cluster distribution"""
    logger.info("\n" + "="*70)
    logger.info("CLUSTER DISTRIBUTION ANALYSIS")
    logger.info("="*70)
    
    cluster_counts = Counter(df['cluster_id'])
    cluster_sizes = np.array(list(cluster_counts.values()))
    
    # Statistics
    logger.info("\n📊 STATISTICS:")
    logger.info(f"   Total documents: {len(df)}")
    logger.info(f"   Number of clusters: {len(cluster_counts)}")
    logger.info(f"   Mean cluster size: {cluster_sizes.mean():.1f}")
    logger.info(f"   Std deviation: {cluster_sizes.std():.1f}")
    logger.info(f"   Min cluster size: {cluster_sizes.min()}")
    logger.info(f"   Max cluster size: {cluster_sizes.max()}")
    logger.info(f"   Imbalance ratio: {cluster_sizes.max()/cluster_sizes.min():.2f}x")
    
    # Distribution by cluster
    logger.info("\n📈 DISTRIBUTION BY CLUSTER:")
    for cluster_id in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster_id]
        percentage = (count / len(df)) * 100
        
        # Get cluster label
        cluster_label = df[df['cluster_id'] == cluster_id]['cluster_label'].iloc[0]
        
        # Status indicator
        if count < 20:
            status = "⚠ UNDERREPRESENTED"
        else:
            status = "✓"
        
        logger.info(f"   Cluster {cluster_id:2d} ({cluster_label:30s}): {count:3d} docs ({percentage:5.1f}%) {status}")
    
    return cluster_counts


def analyze_source_distribution(df):
    """Analyze source distribution"""
    logger.info("\n📚 DISTRIBUTION BY SOURCE:")
    
    source_counts = Counter(df['source'])
    
    for source, count in source_counts.most_common():
        percentage = (count / len(df)) * 100
        logger.info(f"   {source:30s}: {count:3d} docs ({percentage:5.1f}%)")
    
    return source_counts


def analyze_heritage_types(df):
    """Analyze heritage type distribution"""
    logger.info("\n🏛 DISTRIBUTION BY HERITAGE TYPE:")
    
    # Flatten all heritage types
    all_types = []
    for types in df['heritage_types']:
        if isinstance(types, list):
            all_types.extend(types)
    
    type_counts = Counter(all_types)
    
    for htype, count in type_counts.most_common():
        logger.info(f"   {htype:20s}: {count:3d} occurrences")
    
    return type_counts


def analyze_domains(df):
    """Analyze domain distribution"""
    logger.info("\n🎯 DISTRIBUTION BY DOMAIN:")
    
    # Flatten all domains
    all_domains = []
    for domains in df['domains']:
        if isinstance(domains, list):
            all_domains.extend(domains)
    
    domain_counts = Counter(all_domains)
    
    for domain, count in domain_counts.most_common():
        logger.info(f"   {domain:20s}: {count:3d} occurrences")
    
    return domain_counts


def analyze_imbalance(df, cluster_counts):
    """Analyze cluster imbalance and suggest actions"""
    logger.info("\n🔍 IMBALANCE ANALYSIS:")
    
    cluster_sizes = np.array(list(cluster_counts.values()))
    target_size = 35  # Target cluster size
    
    logger.info(f"\n   Target cluster size: {target_size} documents")
    logger.info(f"   Current mean: {cluster_sizes.mean():.1f} documents")
    
    underrepresented = []
    overrepresented = []
    balanced = []
    
    for cluster_id, count in cluster_counts.items():
        diff = target_size - count
        
        if count < 20:  # Significantly underrepresented
            underrepresented.append((cluster_id, count, diff))
        elif count > 45:  # Significantly overrepresented
            overrepresented.append((cluster_id, count, diff))
        else:
            balanced.append((cluster_id, count))
    
    # Report underrepresented clusters
    if underrepresented:
        logger.info(f"\n   ⚠ UNDERREPRESENTED CLUSTERS ({len(underrepresented)}):")
        for cluster_id, count, diff in sorted(underrepresented, key=lambda x: x[1]):
            logger.info(f"      Cluster {cluster_id:2d}: {count:3d} docs (need +{abs(diff)} more)")
    
    # Report overrepresented clusters
    if overrepresented:
        logger.info(f"\n   📈 OVERREPRESENTED CLUSTERS ({len(overrepresented)}):")
        for cluster_id, count, diff in sorted(overrepresented, key=lambda x: x[1], reverse=True):
            logger.info(f"      Cluster {cluster_id:2d}: {count:3d} docs (+{abs(diff)} excess)")
    
    # Report balanced clusters
    if balanced:
        logger.info(f"\n   ✓ BALANCED CLUSTERS ({len(balanced)}):")
        for cluster_id, count in sorted(balanced):
            logger.info(f"      Cluster {cluster_id:2d}: {count:3d} docs (OK)")
    
    imbalance_info = {
        'target_size': target_size,
        'underrepresented': [
            {'cluster_id': int(c), 'count': int(cnt), 'needed': int(abs(d))}
            for c, cnt, d in underrepresented
        ],
        'overrepresented': [
            {'cluster_id': int(c), 'count': int(cnt), 'excess': int(abs(d))}
            for c, cnt, d in overrepresented
        ],
        'balanced': [
            {'cluster_id': int(c), 'count': int(cnt)}
            for c, cnt in balanced
        ]
    }
    
    return imbalance_info


def create_visualizations(df, cluster_counts, source_counts, heritage_counts, domain_counts):
    """Create visualization plots"""
    logger.info("\n📊 Creating visualizations...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (20, 12)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Heritage Documents - Data Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cluster Distribution (Bar)
    ax1 = axes[0, 0]
    clusters = sorted(cluster_counts.keys())
    counts = [cluster_counts[c] for c in clusters]
    colors = ['#e74c3c' if c < 20 else '#2ecc71' if c < 45 else '#f39c12' for c in counts]
    ax1.bar(clusters, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=35, color='blue', linestyle='--', label='Target (35)')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Documents')
    ax1.set_title('Cluster Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Source Distribution (Pie)
    ax2 = axes[0, 1]
    sources = list(source_counts.keys())
    src_counts = list(source_counts.values())
    ax2.pie(src_counts, labels=sources, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Source Distribution')
    
    # 3. Heritage Type Distribution (Horizontal Bar)
    ax3 = axes[0, 2]
    types = [t for t, _ in heritage_counts.most_common(6)]
    type_vals = [heritage_counts[t] for t in types]
    ax3.barh(types, type_vals, color='#3498db', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Occurrences')
    ax3.set_title('Top Heritage Types')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Domain Distribution (Horizontal Bar)
    ax4 = axes[1, 0]
    domains = [d for d, _ in domain_counts.most_common(6)]
    domain_vals = [domain_counts[d] for d in domains]
    ax4.barh(domains, domain_vals, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Occurrences')
    ax4.set_title('Top Domains')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Cluster Size Distribution (Histogram)
    ax5 = axes[1, 1]
    cluster_sizes = list(cluster_counts.values())
    ax5.hist(cluster_sizes, bins=10, color='#1abc9c', alpha=0.7, edgecolor='black')
    ax5.axvline(x=np.mean(cluster_sizes), color='red', linestyle='--', label=f'Mean: {np.mean(cluster_sizes):.1f}')
    ax5.set_xlabel('Cluster Size')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Cluster Size Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics Summary (Text)
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    cluster_sizes_arr = np.array(cluster_sizes)
    stats_text = f"""
    SUMMARY STATISTICS
    ==================
    
    Total Documents: {len(df)}
    Number of Clusters: {len(cluster_counts)}
    
    Cluster Sizes:
      Mean: {cluster_sizes_arr.mean():.1f}
      Std Dev: {cluster_sizes_arr.std():.1f}
      Min: {cluster_sizes_arr.min()}
      Max: {cluster_sizes_arr.max()}
      Imbalance Ratio: {cluster_sizes_arr.max()/cluster_sizes_arr.min():.2f}x
    
    Balance Status:
      Underrepresented: {sum(1 for c in cluster_sizes if c < 20)}
      Balanced: {sum(1 for c in cluster_sizes if 20 <= c <= 45)}
      Overrepresented: {sum(1 for c in cluster_sizes if c > 45)}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(OUTPUT_DIR, 'data_distribution_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✓ Saved: {output_file}")


def save_analysis_report(df, cluster_counts, source_counts, 
                        heritage_counts, domain_counts, 
                        cluster_info, imbalance_info):
    """Save analysis report to JSON"""
    
    # Convert Counter to numpy array for statistics
    cluster_sizes = np.array(list(cluster_counts.values()))
    
    report = {
        'analysis_date': datetime.now().isoformat(),
        'total_documents': int(len(df)),
        'cluster_statistics': {
            'num_clusters': int(len(cluster_counts)),
            'mean_size': float(cluster_sizes.mean()),
            'std_size': float(cluster_sizes.std()),
            'min_size': int(cluster_sizes.min()),
            'max_size': int(cluster_sizes.max()),
            'imbalance_ratio': float(cluster_sizes.max() / cluster_sizes.min())
        },
        'cluster_distribution': {
            int(k): int(v) for k, v in cluster_counts.items()
        },
        'source_distribution': {
            k: int(v) for k, v in source_counts.items()
        },
        'heritage_type_distribution': {
            k: int(v) for k, v in heritage_counts.items()
        },
        'domain_distribution': {
            k: int(v) for k, v in domain_counts.items()
        },
        'cluster_info': cluster_info,
        'imbalance_analysis': imbalance_info
    }
    
    report_file = os.path.join(OUTPUT_DIR, 'data_balance_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 Saved analysis report to: {report_file}")
    
    return report


def main():
    """Main execution function"""
    logger.info("="*70)
    logger.info("DATA BALANCE ANALYSIS")
    logger.info("="*70)
    
    # Load data
    documents = load_classified_data()
    if documents is None:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(documents)
    
    # Extract classifications
    df['heritage_types'] = df['classifications'].apply(lambda x: x.get('heritage_types', []))
    df['domains'] = df['classifications'].apply(lambda x: x.get('domains', []))
    df['time_period'] = df['classifications'].apply(lambda x: x.get('time_period', 'unknown'))
    df['region'] = df['classifications'].apply(lambda x: x.get('region', 'unknown'))
    
    # Analyze cluster distribution
    cluster_counts = analyze_cluster_distribution(df)
    
    # Analyze source distribution
    source_counts = analyze_source_distribution(df)
    
    # Analyze heritage types
    heritage_counts = analyze_heritage_types(df)
    
    # Analyze domains
    domain_counts = analyze_domains(df)
    
    # Analyze imbalance
    imbalance_info = analyze_imbalance(df, cluster_counts)
    
    # Create cluster info
    cluster_info = {}
    for cluster_id in sorted(cluster_counts.keys()):
        cluster_docs = df[df['cluster_id'] == cluster_id]
        cluster_info[str(cluster_id)] = {
            'label': cluster_docs['cluster_label'].iloc[0],
            'size': int(cluster_counts[cluster_id]),
            'percentage': float((cluster_counts[cluster_id] / len(df)) * 100)
        }
    
    # Create visualizations
    create_visualizations(df, cluster_counts, source_counts, heritage_counts, domain_counts)
    
    # Save report
    report = save_analysis_report(df, cluster_counts, source_counts, 
                                  heritage_counts, domain_counts, 
                                  cluster_info, imbalance_info)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Analyzed {len(df)} documents across {len(cluster_counts)} clusters")
    logger.info(f"📊 Visualizations saved to: {OUTPUT_DIR}")
    logger.info(f"📄 Report saved to: {os.path.join(OUTPUT_DIR, 'data_balance_report.json')}")
    logger.info("="*70)


if __name__ == "__main__":
    main()