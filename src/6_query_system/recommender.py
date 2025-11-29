"""
Hybrid Recommender for Heritage Document System

Combines three scoring components:
1. SimRank (0.4 weight): Structural graph similarity
2. Horn's Index (0.3 weight): Entity importance weights
3. Embedding Similarity (0.3 weight): Semantic similarity via FAISS

Provides explainable recommendations with KG path reasoning.
"""

import numpy as np
import pickle
import json
import faiss
from pathlib import Path
from typing import Dict, List, Tuple, Set
import networkx as nx


class HeritageRecommender:
    """Hybrid recommender combining SimRank, Horn's Index, and embeddings."""

    def __init__(
        self,
        kg_path: str = 'knowledge_graph/heritage_kg.gpickle',
        simrank_path: str = 'knowledge_graph/simrank/simrank_matrix.npy',
        embeddings_path: str = 'data/embeddings/document_embeddings.npy',
        metadata_path: str = 'data/embeddings/embedding_mapping.json',
        faiss_index_path: str = 'models/ranker/faiss/hnsw_index.faiss',
        horn_weights_path: str = 'knowledge_graph/horn_weights.json',
        simrank_weight: float = 0.4,
        horn_weight: float = 0.3,
        embedding_weight: float = 0.3
    ):
        """
        Initialize hybrid recommender.

        Args:
            kg_path: Path to knowledge graph pickle file
            simrank_path: Path to SimRank matrix
            embeddings_path: Path to document embeddings
            metadata_path: Path to embedding metadata
            faiss_index_path: Path to FAISS index
            horn_weights_path: Path to Horn's Index weights
            simrank_weight: Weight for SimRank score (default 0.4)
            horn_weight: Weight for Horn's Index score (default 0.3)
            embedding_weight: Weight for embedding similarity (default 0.3)
        """
        print("Loading knowledge graph...")
        with open(kg_path, 'rb') as f:
            self.G = pickle.load(f)

        print("Loading SimRank matrix...")
        self.simrank_matrix = np.load(simrank_path)

        print("Loading embeddings...")
        self.embeddings = np.load(embeddings_path)

        print("Loading metadata...")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(faiss_index_path)

        # Load Horn's weights if available
        self.horn_weights = {}
        if Path(horn_weights_path).exists():
            print("Loading Horn's Index weights...")
            with open(horn_weights_path, 'r') as f:
                self.horn_weights = json.load(f)
        else:
            print("Warning: Horn's weights not found. Using uniform weights.")

        # Scoring weights
        self.simrank_weight = simrank_weight
        self.horn_weight = horn_weight
        self.embedding_weight = embedding_weight

        # Create document index mapping
        self.doc_nodes = [n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'document']
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_nodes)}
        self.idx_to_doc_id = {idx: doc_id for doc_id, idx in self.doc_id_to_idx.items()}

        print(f"Recommender initialized with {len(self.doc_nodes)} documents!")
        print(f"Weights: SimRank={simrank_weight}, Horn={horn_weight}, Embedding={embedding_weight}")

    def compute_simrank_score(self, query_doc_idx: int, candidate_doc_idx: int) -> float:
        """
        Compute SimRank similarity between two documents.

        Args:
            query_doc_idx: Index of query document
            candidate_doc_idx: Index of candidate document

        Returns:
            SimRank similarity score [0, 1]
        """
        if query_doc_idx >= len(self.simrank_matrix) or candidate_doc_idx >= len(self.simrank_matrix):
            return 0.0
        return self.simrank_matrix[query_doc_idx, candidate_doc_idx]

    def compute_horn_score(self, doc_id: str, parsed_query: Dict) -> float:
        """
        Compute Horn's Index score based on entity importance.

        Args:
            doc_id: Document node ID
            parsed_query: Parsed query with extracted entities

        Returns:
            Horn's Index score [0, 1]
        """
        if not self.horn_weights:
            return 0.5  # Uniform weight if Horn's not available

        # Get document's connected entities
        doc_entities = set()
        for neighbor in self.G.neighbors(doc_id):
            neighbor_type = self.G.nodes[neighbor].get('node_type')
            if neighbor_type in ['location', 'person', 'organization', 'heritage_type', 'domain', 'time_period', 'region']:
                doc_entities.add(neighbor)

        # Convert query entities to KG entity IDs (with prefixes)
        query_entities = set()

        # Heritage types: "temple" -> "type_temple"
        for heritage_type in parsed_query.get('heritage_types', []):
            query_entities.add(f"type_{heritage_type}")

        # Domains: "religious" -> "domain_religious"
        for domain in parsed_query.get('domains', []):
            query_entities.add(f"domain_{domain}")

        # Time period: "ancient" -> "period_ancient"
        if parsed_query.get('time_period'):
            query_entities.add(f"period_{parsed_query['time_period']}")

        # Region: "north" -> "region_north"
        if parsed_query.get('region'):
            query_entities.add(f"region_{parsed_query['region']}")

        # Locations, persons, orgs: "India" -> "loc_india" (lowercase)
        for location in parsed_query.get('locations', []):
            query_entities.add(f"loc_{location.lower().replace(' ', '_')}")
        for person in parsed_query.get('persons', []):
            query_entities.add(f"person_{person.lower().replace(' ', '_')}")
        for org in parsed_query.get('organizations', []):
            query_entities.add(f"org_{org.lower().replace(' ', '_')}")

        if not query_entities or not doc_entities:
            return 0.0

        # Compute weighted overlap
        matching_entities = doc_entities.intersection(query_entities)
        if not matching_entities:
            return 0.0

        # Sum of Horn weights for matching entities
        total_weight = sum(self.horn_weights.get(entity, 0.5) for entity in matching_entities)
        max_possible_weight = len(matching_entities)  # Max weight if all were 1.0

        return total_weight / max_possible_weight if max_possible_weight > 0 else 0.0

    def compute_embedding_similarity(self, query_embedding: np.ndarray, doc_idx: int) -> float:
        """
        Compute cosine similarity between query and document embeddings.

        Args:
            query_embedding: Query embedding vector (384-dim)
            doc_idx: Document index in embeddings matrix

        Returns:
            Cosine similarity score [0, 1]
        """
        if doc_idx >= len(self.embeddings):
            return 0.0

        doc_embedding = self.embeddings[doc_idx]

        # Cosine similarity (both are L2-normalized)
        similarity = np.dot(query_embedding, doc_embedding)

        # Ensure [0, 1] range
        return max(0.0, min(1.0, similarity))

    def get_kg_path_explanation(self, source_doc_id: str, target_doc_id: str, max_paths: int = 3) -> List[List[str]]:
        """
        Find shortest paths between documents in KG for explanation.

        Args:
            source_doc_id: Source document node ID
            target_doc_id: Target document node ID
            max_paths: Maximum number of paths to return

        Returns:
            List of paths (each path is a list of node IDs)
        """
        try:
            # Find all simple paths up to length 4
            paths = list(nx.all_simple_paths(
                self.G,
                source=source_doc_id,
                target=target_doc_id,
                cutoff=4
            ))

            # Sort by length and return top paths
            paths = sorted(paths, key=len)[:max_paths]
            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def format_path_explanation(self, path: List[str]) -> str:
        """
        Format a KG path into human-readable explanation.

        Args:
            path: List of node IDs in path

        Returns:
            Formatted path string
        """
        formatted_parts = []
        for node_id in path:
            node_data = self.G.nodes[node_id]
            node_type = node_data.get('node_type', 'unknown')

            if node_type == 'document':
                formatted_parts.append(f"[{node_data.get('title', node_id)}]")
            else:
                formatted_parts.append(f"({node_type}: {node_id})")

        return " � ".join(formatted_parts)

    def recommend(
        self,
        parsed_query: Dict,
        top_k: int = 10,
        explain: bool = True
    ) -> List[Dict]:
        """
        Generate top-K recommendations using hybrid scoring.

        Args:
            parsed_query: Parsed query from QueryProcessor
            top_k: Number of recommendations to return
            explain: Whether to include KG path explanations

        Returns:
            List of recommendation dictionaries with scores and explanations
        """
        query_embedding = parsed_query['query_embedding']
        scores = []

        # Compute hybrid scores for all documents
        for doc_idx, doc_id in enumerate(self.doc_nodes):
            # Embedding similarity (always available)
            emb_score = self.compute_embedding_similarity(query_embedding, doc_idx)

            # SimRank score (structural similarity)
            # For new queries, use average SimRank with top-k similar docs
            top_k_similar_indices = self._get_top_k_similar_by_embedding(query_embedding, k=5)
            simrank_scores = [self.compute_simrank_score(similar_idx, doc_idx)
                            for similar_idx in top_k_similar_indices]
            simrank_score = np.mean(simrank_scores) if simrank_scores else 0.0

            # Horn's Index score (entity importance)
            horn_score = self.compute_horn_score(doc_id, parsed_query)

            # Hybrid score
            hybrid_score = (
                self.simrank_weight * simrank_score +
                self.horn_weight * horn_score +
                self.embedding_weight * emb_score
            )

            scores.append({
                'doc_id': doc_id,
                'doc_idx': doc_idx,
                'hybrid_score': hybrid_score,
                'simrank_score': simrank_score,
                'horn_score': horn_score,
                'embedding_score': emb_score
            })

        # Sort by hybrid score
        scores = sorted(scores, key=lambda x: x['hybrid_score'], reverse=True)

        # Get top-K recommendations
        recommendations = []
        for rank, score_dict in enumerate(scores[:top_k], 1):
            doc_id = score_dict['doc_id']
            doc_data = self.G.nodes[doc_id]

            rec = {
                'rank': rank,
                'doc_id': doc_id,
                'title': doc_data.get('title', 'Unknown'),
                'hybrid_score': score_dict['hybrid_score'],
                'component_scores': {
                    'simrank': score_dict['simrank_score'],
                    'horn': score_dict['horn_score'],
                    'embedding': score_dict['embedding_score']
                },
                'metadata': {
                    'heritage_type': doc_data.get('heritage_type'),
                    'domain': doc_data.get('domain'),
                    'time_period': doc_data.get('time_period'),
                    'region': doc_data.get('region')
                }
            }

            # Add KG path explanations
            if explain:
                # Find paths to top similar documents
                top_similar_docs = [scores[i]['doc_id'] for i in range(min(3, len(scores)))
                                  if scores[i]['doc_id'] != doc_id]

                explanations = []
                for similar_doc_id in top_similar_docs:
                    paths = self.get_kg_path_explanation(similar_doc_id, doc_id, max_paths=1)
                    if paths:
                        path_str = self.format_path_explanation(paths[0])
                        explanations.append(path_str)

                rec['kg_explanations'] = explanations

            recommendations.append(rec)

        return recommendations

    def _get_top_k_similar_by_embedding(self, query_embedding: np.ndarray, k: int = 5) -> List[int]:
        """
        Get top-K most similar documents by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            k: Number of similar docs to retrieve

        Returns:
            List of document indices
        """
        # FAISS search
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k)
        return indices[0].tolist()

    def format_recommendation(self, rec: Dict) -> str:
        """
        Format a recommendation for display.

        Args:
            rec: Recommendation dictionary

        Returns:
            Formatted string
        """
        lines = [
            f"#{rec['rank']} {rec['title']}",
            f"  Score: {rec['hybrid_score']:.4f} (SimRank: {rec['component_scores']['simrank']:.3f}, "
            f"Horn: {rec['component_scores']['horn']:.3f}, Embedding: {rec['component_scores']['embedding']:.3f})",
        ]

        # Metadata
        meta = rec['metadata']
        meta_parts = []
        if meta.get('heritage_type'):
            meta_parts.append(f"Type: {meta['heritage_type']}")
        if meta.get('domain'):
            meta_parts.append(f"Domain: {meta['domain']}")
        if meta.get('time_period'):
            meta_parts.append(f"Period: {meta['time_period']}")
        if meta.get('region'):
            meta_parts.append(f"Region: {meta['region']}")

        if meta_parts:
            lines.append(f"  {' | '.join(meta_parts)}")

        # KG explanations
        if rec.get('kg_explanations'):
            lines.append("  Why recommended:")
            for explanation in rec['kg_explanations'][:2]:  # Show top 2 paths
                lines.append(f"    • {explanation}")

        return '\n'.join(lines)


def main():
    """Test recommender with sample parsed queries."""
    from query_processor import QueryProcessor

    # Initialize
    processor = QueryProcessor()
    recommender = HeritageRecommender()

    # Test queries
    test_queries = [
        "Mughal temples in North India",
        "Ancient forts in Rajasthan",
        "Buddhist stupas and monasteries"
    ]

    print("\n" + "=" * 80)
    print("HERITAGE RECOMMENDER TEST")
    print("=" * 80)

    for query_text in test_queries:
        print(f"\nQuery: {query_text}")
        print("-" * 80)

        # Parse query
        parsed_query = processor.parse_query(query_text)

        # Get recommendations
        recommendations = recommender.recommend(parsed_query, top_k=5, explain=True)

        # Display results
        for rec in recommendations:
            print(recommender.format_recommendation(rec))
            print()


if __name__ == '__main__':
    main()
