"""
Data Balancing & Augmentation
Use SMOTE + NLP techniques to balance dataset across clusters
"""

import json
import os
import sys
import random
import numpy as np
from collections import defaultdict
from datetime import datetime
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config_loader import get_config
from utils.logger import get_logger

# Initialize
config = get_config()
logger = get_logger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class DataAugmenter:
    """Data augmentation using NLP techniques"""
    
    def __init__(self):
        logger.info("Initializing data augmenters...")
        
        # Synonym replacement
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=5)
        
        # Contextual word embeddings (BERT-based)
        try:
            self.contextual_aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased',
                action="substitute",
                aug_min=1,
                aug_max=5
            )
        except Exception as e:
            logger.warning(f"Could not load contextual augmenter: {e}")
            self.contextual_aug = None
        
        # Back-translation (if you have internet)
        try:
            self.back_trans_aug = naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de',
                to_model_name='facebook/wmt19-de-en',
                device='cpu'
            )
        except Exception as e:
            logger.warning(f"Could not load back-translation: {e}")
            self.back_trans_aug = None
        
        logger.info("✓ Augmenters ready")
    
    def augment_text(self, text, method='synonym', num_variants=1):
        """
        Augment text using specified method
        
        Args:
            text: Input text
            method: 'synonym', 'contextual', or 'back_translation'
            num_variants: Number of augmented versions to generate
        
        Returns:
            List of augmented texts
        """
        augmented = []
        
        # Take first 1000 characters for augmentation (speed)
        text_snippet = text[:1000]
        
        try:
            if method == 'synonym':
                for _ in range(num_variants):
                    aug_text = self.synonym_aug.augment(text_snippet)
                    if isinstance(aug_text, list):
                        aug_text = aug_text[0]
                    augmented.append(aug_text)
            
            elif method == 'contextual' and self.contextual_aug:
                for _ in range(num_variants):
                    aug_text = self.contextual_aug.augment(text_snippet)
                    if isinstance(aug_text, list):
                        aug_text = aug_text[0]
                    augmented.append(aug_text)
            
            elif method == 'back_translation' and self.back_trans_aug:
                aug_text = self.back_trans_aug.augment(text_snippet)
                if isinstance(aug_text, list):
                    aug_text = aug_text[0]
                augmented.append(aug_text)
            
            else:
                # Fallback: simple synonym replacement
                augmented.append(self.synonym_aug.augment(text_snippet))
        
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            # Return original text with slight modification
            augmented.append(text_snippet)
        
        return augmented

def load_data():
    """Load classified documents and embeddings"""
    logger.info("Loading data...")
    
    # Load classified documents
    classified_file = config.get_path('data', 'classified') + '/classified_documents.json'
    with open(classified_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Load embeddings
    embeddings_file = config.get_path('data', 'embeddings') + '/document_embeddings.npy'
    embeddings = np.load(embeddings_file)
    
    # Load cleaned texts
    texts = []
    for doc in documents:
        cleaned_path = doc.get('cleaned_path', '')
        if os.path.exists(cleaned_path):
            with open(cleaned_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        else:
            texts.append(doc.get('summary', ''))
    
    logger.info(f"✓ Loaded {len(documents)} documents")
    logger.info(f"✓ Loaded embeddings: {embeddings.shape}")
    
    return documents, embeddings, texts

def analyze_cluster_balance(documents):
    """Analyze cluster distribution"""
    cluster_groups = defaultdict(list)
    
    for idx, doc in enumerate(documents):
        cluster_groups[doc['cluster_id']].append(idx)
    
    cluster_sizes = {cid: len(indices) for cid, indices in cluster_groups.items()}
    
    return cluster_groups, cluster_sizes

def balance_with_smote(embeddings, cluster_labels, target_size):
    """
    Balance embeddings using SMOTE
    
    Args:
        embeddings: Original embeddings
        cluster_labels: Cluster IDs for each document
        target_size: Target number of samples per cluster
    
    Returns:
        Synthetic embeddings, their cluster labels, original indices
    """
    logger.info("\n[SMOTE] Generating synthetic embeddings...")
    
    # Group by cluster
    cluster_groups = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        cluster_groups[label].append(idx)
    
    synthetic_embeddings = []
    synthetic_labels = []
    synthetic_sources = []  # Track which original doc was used
    
    for cluster_id, indices in cluster_groups.items():
        current_size = len(indices)
        
        if current_size >= target_size:
            logger.info(f"   Cluster {cluster_id}: {current_size} docs (already balanced)")
            continue
        
        deficit = target_size - current_size
        logger.info(f"   Cluster {cluster_id}: {current_size} docs → need {deficit} more")
        
        # Get embeddings for this cluster
        cluster_embeddings = embeddings[indices]
        
        if current_size < 2:
            # Not enough samples for SMOTE, duplicate + add noise
            logger.warning(f"      Too few samples for SMOTE, using duplication with noise")
            for _ in range(deficit):
                # Pick random sample and add Gaussian noise
                orig_idx = random.choice(indices)
                noisy_embedding = embeddings[orig_idx] + np.random.normal(0, 0.02, embeddings.shape[1])
                # Normalize to maintain unit norm
                noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
                synthetic_embeddings.append(noisy_embedding)
                synthetic_labels.append(cluster_id)
                synthetic_sources.append(orig_idx)
        else:
            # Use SMOTE with more aggressive sampling
            try:
                # For small clusters, use all neighbors available
                k_neighbors = min(5, current_size - 1)
                
                # Create more synthetic samples than needed, then sample
                oversample_ratio = min(3.0, deficit / current_size)
                
                smote = SMOTE(
                    sampling_strategy={1: int(current_size * (1 + oversample_ratio))},
                    k_neighbors=k_neighbors, 
                    random_state=42
                )
                
                # Create binary labels for SMOTE
                y_dummy = np.zeros(current_size)
                y_dummy[:1] = 1  # Just need one positive class
                
                # Resample
                X_resampled, _ = smote.fit_resample(cluster_embeddings, y_dummy)
                
                # Take synthetic samples (skip original ones)
                synthetic_batch = X_resampled[current_size:]
                
                # Take up to deficit samples
                num_to_take = min(len(synthetic_batch), deficit)
                selected_samples = synthetic_batch[:num_to_take]
                
                for synthetic_emb in selected_samples:
                    synthetic_embeddings.append(synthetic_emb)
                    synthetic_labels.append(cluster_id)
                    # Assign to nearest original document
                    distances = np.linalg.norm(cluster_embeddings - synthetic_emb, axis=1)
                    nearest_idx = indices[np.argmin(distances)]
                    synthetic_sources.append(nearest_idx)
                
                logger.info(f"      ✓ Generated {len(selected_samples)} synthetic samples via SMOTE")
                
                # If still need more, duplicate with noise
                remaining = deficit - len(selected_samples)
                if remaining > 0:
                    logger.info(f"      Adding {remaining} more via duplication...")
                    for _ in range(remaining):
                        orig_idx = random.choice(indices)
                        noisy_embedding = embeddings[orig_idx] + np.random.normal(0, 0.02, embeddings.shape[1])
                        noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
                        synthetic_embeddings.append(noisy_embedding)
                        synthetic_labels.append(cluster_id)
                        synthetic_sources.append(orig_idx)
            
            except Exception as e:
                logger.warning(f"      SMOTE failed: {e}, using duplication")
                for _ in range(deficit):
                    orig_idx = random.choice(indices)
                    noisy_embedding = embeddings[orig_idx] + np.random.normal(0, 0.02, embeddings.shape[1])
                    noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
                    synthetic_embeddings.append(noisy_embedding)
                    synthetic_labels.append(cluster_id)
                    synthetic_sources.append(orig_idx)
    
    if synthetic_embeddings:
        synthetic_embeddings = np.array(synthetic_embeddings)
        logger.info(f"\n   ✓ Generated {len(synthetic_embeddings)} synthetic embeddings")
        return synthetic_embeddings, synthetic_labels, synthetic_sources
    else:
        logger.info(f"\n   ℹ No synthetic embeddings needed")
        return None, None, None

def create_augmented_documents(documents, texts, synthetic_labels, synthetic_sources, augmenter):
    """Create augmented document metadata"""
    logger.info("\n[NLP Augmentation] Creating augmented documents...")
    
    augmented_docs = []
    augmented_texts = []
    
    augmentation_methods = ['synonym', 'contextual', 'synonym']  # Cycle through methods
    
    for idx, (cluster_id, source_idx) in enumerate(zip(synthetic_labels, synthetic_sources)):
        source_doc = documents[source_idx]
        source_text = texts[source_idx]
        
        # Choose augmentation method
        method = augmentation_methods[idx % len(augmentation_methods)]
        
        # Augment text
        try:
            aug_texts = augmenter.augment_text(source_text, method=method, num_variants=1)
            aug_text = aug_texts[0] if aug_texts else source_text
        except Exception as e:
            logger.warning(f"   Augmentation failed for doc {source_idx}: {e}")
            aug_text = source_text  # Fallback to original
        
        # Create augmented document metadata
        aug_doc = {
            **source_doc,  # Copy all metadata
            'title': f"{source_doc['title']} (Augmented-{idx+1})",
            'file_name': f"augmented_{cluster_id}_{idx+1}.txt",
            'source': f"{source_doc['source']} (Synthetic)",
            'is_synthetic': True,
            'augmentation_method': method,
            'source_document_index': source_idx,
            'synthetic_index': idx,
            'created_at': datetime.now().isoformat()
        }
        
        augmented_docs.append(aug_doc)
        augmented_texts.append(aug_text)
    
    logger.info(f"   ✓ Created {len(augmented_docs)} augmented documents")
    
    return augmented_docs, augmented_texts

def save_balanced_dataset(original_docs, original_embeddings, original_texts,
                          augmented_docs, augmented_embeddings, augmented_texts):
    """Save balanced dataset"""
    logger.info("\n[Saving] Creating balanced dataset...")
    
    output_dir = config.get_path('data', 'balanced')
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine original + augmented
    all_docs = original_docs + augmented_docs
    all_embeddings = np.vstack([original_embeddings, augmented_embeddings]) if augmented_embeddings is not None else original_embeddings
    all_texts = original_texts + augmented_texts
    
    # Save balanced documents
    balanced_file = os.path.join(output_dir, 'balanced_documents.json')
    with open(balanced_file, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    logger.info(f"   ✓ Saved: {balanced_file}")
    
    # Save balanced embeddings
    embeddings_file = os.path.join(output_dir, 'balanced_embeddings.npy')
    np.save(embeddings_file, all_embeddings)
    logger.info(f"   ✓ Saved: {embeddings_file}")
    
    # Save augmented texts
    texts_dir = os.path.join(output_dir, 'texts')
    os.makedirs(texts_dir, exist_ok=True)
    
    for doc, text in zip(augmented_docs, augmented_texts):
        text_file = os.path.join(texts_dir, doc['file_name'])
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
    
    logger.info(f"   ✓ Saved {len(augmented_texts)} augmented texts to: {texts_dir}")
    
    # Save balancing report
    cluster_dist = defaultdict(int)
    for doc in all_docs:
        cluster_dist[doc['cluster_id']] += 1
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'original_size': len(original_docs),
        'augmented_size': len(augmented_docs),
        'total_size': len(all_docs),
        'cluster_distribution': dict(cluster_dist),
        'augmentation_summary': {
            'synthetic_samples': len(augmented_docs),
            'methods_used': ['SMOTE', 'synonym_replacement', 'contextual_substitution']
        }
    }
    
    report_file = os.path.join(output_dir, 'balancing_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"   ✓ Saved report: {report_file}")
    
    return all_docs, all_embeddings, report

def main():
    logger.info("="*70)
    logger.info("DATA BALANCING & AUGMENTATION")
    logger.info("="*70)
    
    # Configuration
    target_size = config.get('balancing', 'target_samples_per_cluster', default=35)
    logger.info(f"\nTarget cluster size: {target_size} documents")
    
    # Load data
    documents, embeddings, texts = load_data()
    
    # Analyze current balance
    cluster_groups, cluster_sizes = analyze_cluster_balance(documents)
    logger.info(f"\nCurrent cluster distribution: {cluster_sizes}")
    
    # Get cluster labels
    cluster_labels = np.array([doc['cluster_id'] for doc in documents])
    
    # Balance with SMOTE
    synthetic_embeddings, synthetic_labels, synthetic_sources = balance_with_smote(
        embeddings, cluster_labels, target_size
    )
    
    if synthetic_embeddings is None:
        logger.info("\n✓ Dataset already balanced!")
        return
    
    # Initialize augmenter
    augmenter = DataAugmenter()
    
    # Create augmented documents
    augmented_docs, augmented_texts = create_augmented_documents(
        documents, texts, synthetic_labels, synthetic_sources, augmenter
    )
    
    # Save balanced dataset
    all_docs, all_embeddings, report = save_balanced_dataset(
        documents, embeddings, texts,
        augmented_docs, synthetic_embeddings, augmented_texts
    )
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("BALANCING COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Original documents: {len(documents)}")
    logger.info(f"✅ Augmented documents: {len(augmented_docs)}")
    logger.info(f"✅ Total balanced dataset: {len(all_docs)}")
    logger.info(f"\n📊 New cluster distribution:")
    for cluster_id_str in sorted(report['cluster_distribution'].keys(), key=lambda x: int(x)):
        count = report['cluster_distribution'][cluster_id_str]
        logger.info(f"   Cluster {cluster_id_str}: {count} documents")
    logger.info("="*70)

if __name__ == "__main__":
    main()