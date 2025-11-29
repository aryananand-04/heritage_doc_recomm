"""
Domain-Specific Semantic Similarity System

Combines heritage ontology, embeddings, and manual similarity matrices
to compute accurate semantic similarity for heritage domain concepts.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from heritage_ontology import HeritageOntology, create_default_ontology


class HeritageSimilarity:
    """
    Heritage domain semantic similarity system.

    Combines three approaches:
    1. Ontology-based similarity (heritage_ontology.py)
    2. Embedding-based similarity (domain-specific or general)
    3. Manual similarity matrix for core concepts
    """

    def __init__(self, ontology: Optional[HeritageOntology] = None,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        self.ontology = ontology or create_default_ontology()
        self.embedding_model = SentenceTransformer(embedding_model)

        # Manual similarity matrix for core heritage concepts
        self.manual_similarity = self._build_manual_similarity_matrix()

        # Embedding cache
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def _build_manual_similarity_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Build manual similarity matrix for 100 most common heritage terms.

        This overrides default similarity with domain expert knowledge.
        """

        similarity_pairs = {}

        # Heritage types (high similarity within category)
        heritage_type_groups = [
            ('monument', 'architecture', 0.9),
            ('monument', 'site', 0.8),
            ('monument', 'memorial', 0.95),
            ('architecture', 'building', 0.95),
            ('architecture', 'structure', 0.9),
            ('site', 'place', 0.85),
            ('site', 'location', 0.8),
            ('site', 'complex', 0.75),
            ('artifact', 'relic', 0.95),
            ('artifact', 'object', 0.85),
            ('art', 'artwork', 0.95),
            ('art', 'sculpture', 0.85),
            ('art', 'carving', 0.8),
            ('tradition', 'custom', 0.9),
            ('tradition', 'ritual', 0.85),
            ('tradition', 'practice', 0.9),
        ]

        # Monument types and architecture
        monument_types = [
            ('temple', 'shrine', 0.85),
            ('temple', 'monastery', 0.7),
            ('mosque', 'masjid', 0.98),
            ('mosque', 'dargah', 0.7),
            ('fort', 'fortress', 0.95),
            ('fort', 'citadel', 0.9),
            ('fort', 'castle', 0.85),
            ('palace', 'castle', 0.8),
            ('palace', 'fort', 0.75),
            ('tomb', 'mausoleum', 0.95),
            ('tomb', 'memorial', 0.85),
            ('stupa', 'pagoda', 0.85),
            ('stupa', 'monument', 0.75),
            ('tower', 'minaret', 0.85),
            ('gate', 'gateway', 0.98),
            ('gate', 'arch', 0.8),
        ]

        # Religious terms
        religious_terms = [
            ('hindu', 'hinduism', 0.98),
            ('buddhist', 'buddhism', 0.98),
            ('islamic', 'islam', 0.98),
            ('jain', 'jainism', 0.98),
            ('sikh', 'sikhism', 0.98),
            ('temple', 'mandir', 0.95),
            ('temple', 'pagoda', 0.75),
            ('mosque', 'shrine', 0.7),
            ('monastery', 'vihara', 0.9),
            ('monastery', 'abbey', 0.85),
        ]

        # Architectural styles
        architectural_styles = [
            ('indo-islamic', 'mughal', 0.85),
            ('indo-islamic', 'sultanate', 0.8),
            ('mughal', 'moghul', 0.98),
            ('dravidian', 'south indian', 0.9),
            ('nagara', 'north indian', 0.9),
            ('dravidian', 'nagara', 0.7),  # Related but distinct
            ('vesara', 'dravidian', 0.75),
            ('vesara', 'nagara', 0.75),
            ('rock-cut', 'cave', 0.9),
            ('rock-cut', 'carved', 0.85),
            ('colonial', 'british', 0.85),
            ('colonial', 'european', 0.8),
        ]

        # Time periods
        time_periods = [
            ('ancient', 'early', 0.8),
            ('ancient', 'classical', 0.85),
            ('medieval', 'middle ages', 0.95),
            ('modern', 'contemporary', 0.85),
            ('modern', 'recent', 0.8),
            ('prehistoric', 'ancient', 0.75),
        ]

        # Regions
        regions = [
            ('north india', 'northern', 0.9),
            ('south india', 'southern', 0.9),
            ('east india', 'eastern', 0.9),
            ('west india', 'western', 0.9),
            ('central india', 'central', 0.9),
            ('india', 'indian', 0.95),
            ('india', 'bharat', 0.95),
        ]

        # Dynasties and empires
        dynasties = [
            ('mughal', 'timurid', 0.85),
            ('mauryan', 'maurya', 0.98),
            ('gupta', 'guptas', 0.98),
            ('chola', 'cholas', 0.98),
            ('vijayanagara', 'vijayanagar', 0.98),
            ('maratha', 'marathas', 0.98),
            ('sultanate', 'sultans', 0.9),
        ]

        # Heritage-related actions
        heritage_actions = [
            ('built', 'constructed', 0.95),
            ('built', 'erected', 0.9),
            ('carved', 'sculpted', 0.9),
            ('carved', 'engraved', 0.85),
            ('restored', 'renovated', 0.9),
            ('preserved', 'conserved', 0.95),
            ('excavated', 'unearthed', 0.9),
        ]

        # Heritage-related descriptors
        descriptors = [
            ('historical', 'historic', 0.98),
            ('cultural', 'heritage', 0.85),
            ('archaeological', 'ancient', 0.75),
            ('architectural', 'building', 0.8),
            ('sacred', 'holy', 0.95),
            ('sacred', 'religious', 0.9),
            ('royal', 'imperial', 0.9),
            ('royal', 'regal', 0.95),
            ('military', 'defense', 0.85),
            ('military', 'fortification', 0.85),
        ]

        # Cross-category semantic relationships (lower similarity)
        cross_category = [
            ('temple', 'religious', 0.6),
            ('fort', 'military', 0.6),
            ('palace', 'royal', 0.6),
            ('tomb', 'memorial', 0.7),
            ('monastery', 'buddhist', 0.6),
            ('mosque', 'islamic', 0.6),
        ]

        # Compile all pairs (symmetric)
        all_pairs = (
            heritage_type_groups + monument_types + religious_terms +
            architectural_styles + time_periods + regions +
            dynasties + heritage_actions + descriptors + cross_category
        )

        for term1, term2, score in all_pairs:
            similarity_pairs[(term1.lower(), term2.lower())] = score
            similarity_pairs[(term2.lower(), term1.lower())] = score  # Symmetric

        return similarity_pairs

    def compute_similarity(self, term1: str, term2: str,
                           method: str = 'hybrid') -> float:
        """
        Compute semantic similarity between two terms.

        Args:
            term1: First term
            term2: Second term
            method: 'ontology', 'embedding', 'manual', or 'hybrid'

        Returns:
            Similarity score [0, 1]
        """

        term1_lower = term1.lower().strip()
        term2_lower = term2.lower().strip()

        # Exact match
        if term1_lower == term2_lower:
            return 1.0

        if method == 'manual' or method == 'hybrid':
            # Check manual similarity matrix
            manual_score = self.manual_similarity.get((term1_lower, term2_lower))
            if manual_score is not None:
                if method == 'manual':
                    return manual_score
                # For hybrid, use manual as base and boost with others
                ontology_score = self.ontology.compute_semantic_similarity(term1, term2)
                return max(manual_score, ontology_score)

        if method == 'ontology' or method == 'hybrid':
            # Use heritage ontology
            ontology_score = self.ontology.compute_semantic_similarity(term1, term2)
            if ontology_score > 0.0:
                return ontology_score

        if method == 'embedding' or method == 'hybrid':
            # Use embedding similarity
            return self._embedding_similarity(term1, term2)

        return 0.0

    def _embedding_similarity(self, term1: str, term2: str) -> float:
        """Compute embedding-based cosine similarity"""

        # Get embeddings (with caching)
        if term1 not in self.embedding_cache:
            self.embedding_cache[term1] = self.embedding_model.encode(term1, convert_to_numpy=True)

        if term2 not in self.embedding_cache:
            self.embedding_cache[term2] = self.embedding_model.encode(term2, convert_to_numpy=True)

        emb1 = self.embedding_cache[term1]
        emb2 = self.embedding_cache[term2]

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    def get_similar_concepts(self, term: str, threshold: float = 0.6,
                             top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar concepts from ontology and manual matrix.

        Args:
            term: Query term
            threshold: Minimum similarity threshold
            top_k: Maximum number of results

        Returns:
            List of (concept, similarity) tuples
        """

        candidates = []

        # Get related entities from ontology
        canonical = self.ontology.link_entity(term)
        if canonical:
            related = self.ontology.get_related_entities(canonical)
            for rel in related:
                sim = self.compute_similarity(term, rel, method='ontology')
                if sim >= threshold:
                    candidates.append((rel, sim))

        # Check manual similarity matrix
        term_lower = term.lower()
        for (t1, t2), sim in self.manual_similarity.items():
            if t1 == term_lower and sim >= threshold:
                candidates.append((t2, sim))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    def save_similarity_matrix(self, filepath: str):
        """Save manual similarity matrix to JSON"""
        # Convert tuple keys to string keys for JSON
        json_matrix = {
            f"{k[0]}__{k[1]}": v
            for k, v in self.manual_similarity.items()
        }

        with open(filepath, 'w') as f:
            json.dump(json_matrix, f, indent=2)

    @classmethod
    def load_similarity_matrix(cls, filepath: str, ontology: Optional[HeritageOntology] = None) -> 'HeritageSimilarity':
        """Load similarity matrix from JSON"""
        with open(filepath, 'r') as f:
            json_matrix = json.load(f)

        similarity = cls(ontology=ontology)

        # Convert string keys back to tuple keys
        similarity.manual_similarity = {
            tuple(k.split('__')): v
            for k, v in json_matrix.items()
        }

        return similarity


def compute_concept_similarity_batch(concepts: List[str],
                                      similarity_system: HeritageSimilarity,
                                      threshold: float = 0.5) -> List[Tuple[str, str, float]]:
    """
    Compute pairwise similarity for batch of concepts.

    Args:
        concepts: List of concept strings
        similarity_system: Heritage similarity system
        threshold: Minimum similarity threshold

    Returns:
        List of (concept1, concept2, similarity) tuples above threshold
    """

    edges = []

    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            sim = similarity_system.compute_similarity(
                concepts[i],
                concepts[j],
                method='hybrid'
            )

            if sim >= threshold:
                edges.append((concepts[i], concepts[j], sim))

    return edges


if __name__ == "__main__":
    # Test heritage similarity system
    print("=" * 80)
    print("HERITAGE SEMANTIC SIMILARITY SYSTEM")
    print("=" * 80)

    # Create similarity system
    similarity = HeritageSimilarity()

    # Test cases
    test_pairs = [
        ('temple', 'shrine'),
        ('fort', 'fortress'),
        ('monument', 'architecture'),
        ('mughal', 'moghul'),
        ('dravidian', 'nagara'),
        ('ancient', 'medieval'),
        ('temple', 'mosque'),
        ('taj mahal', 'red fort'),
    ]

    print("\n📊 Similarity Scores (Hybrid Method):\n")
    for term1, term2 in test_pairs:
        score = similarity.compute_similarity(term1, term2, method='hybrid')
        print(f"  {term1:20s} ↔ {term2:20s}: {score:.3f}")

    print("\n📊 Similar Concepts:\n")
    test_terms = ['temple', 'fort', 'mughal', 'ancient']
    for term in test_terms:
        similar = similarity.get_similar_concepts(term, threshold=0.6, top_k=5)
        print(f"  {term}:")
        for concept, sim in similar:
            print(f"    → {concept}: {sim:.3f}")

    # Save similarity matrix
    Path("data/ontology").mkdir(parents=True, exist_ok=True)
    similarity.save_similarity_matrix("data/ontology/similarity_matrix.json")
    print(f"\n✓ Saved similarity matrix to data/ontology/similarity_matrix.json")

    print(f"\n📈 Statistics:")
    print(f"  Manual similarity pairs: {len(similarity.manual_similarity) // 2}")  # Divided by 2 for symmetric
    print(f"  Ontology entities: {len(similarity.ontology.entities)}")
    print(f"  Embedding model: {similarity.embedding_model}")
