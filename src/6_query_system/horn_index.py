"""
Horn's Index for Entity Importance Weighting

Computes importance weights for entities (locations, persons, organizations,
heritage types, domains) based on their structural properties in the KG:

1. Degree centrality: How many connections an entity has
2. Betweenness centrality: How often entity appears on shortest paths
3. PageRank: Importance based on link structure
4. Document frequency: How many documents mention the entity

Horn's Index combines these metrics to identify "important" entities that
should receive higher weight in recommendation scoring.

Formula:
    Horn(entity) = α * degree_norm + β * betweenness_norm + γ * pagerank_norm + δ * df_norm

Where:
    α, β, γ, δ are weights (default: 0.3, 0.2, 0.3, 0.2)
    *_norm are min-max normalized scores [0, 1]
"""

import pickle
import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Set


class HornIndexCalculator:
    """Calculates Horn's Index for entity importance in knowledge graph."""

    def __init__(
        self,
        kg_path: str = 'knowledge_graph/heritage_kg.gpickle',
        degree_weight: float = 0.3,
        betweenness_weight: float = 0.2,
        pagerank_weight: float = 0.3,
        df_weight: float = 0.2
    ):
        """
        Initialize Horn's Index calculator.

        Args:
            kg_path: Path to knowledge graph pickle file
            degree_weight: Weight for degree centrality (default 0.3)
            betweenness_weight: Weight for betweenness centrality (default 0.2)
            pagerank_weight: Weight for PageRank (default 0.3)
            df_weight: Weight for document frequency (default 0.2)
        """
        print(f"Loading knowledge graph from {kg_path}...")
        with open(kg_path, 'rb') as f:
            self.G = pickle.load(f)

        self.degree_weight = degree_weight
        self.betweenness_weight = betweenness_weight
        self.pagerank_weight = pagerank_weight
        self.df_weight = df_weight

        # Entity types to compute weights for
        self.entity_types = {'location', 'person', 'organization', 'heritage_type', 'domain', 'time_period', 'region'}

        print(f"KG loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def get_entities(self) -> Set[str]:
        """
        Get all entity nodes from the graph.

        Returns:
            Set of entity node IDs
        """
        entities = set()
        for node, data in self.G.nodes(data=True):
            if data.get('node_type') in self.entity_types:
                entities.add(node)
        return entities

    def compute_degree_centrality(self, entities: Set[str]) -> Dict[str, float]:
        """
        Compute degree centrality for entities.

        Args:
            entities: Set of entity node IDs

        Returns:
            Dictionary mapping entity ID to degree centrality score
        """
        print("Computing degree centrality...")
        degree_centrality = nx.degree_centrality(self.G)

        # Filter to entities only
        entity_degrees = {e: degree_centrality[e] for e in entities}

        return entity_degrees

    def compute_betweenness_centrality(self, entities: Set[str]) -> Dict[str, float]:
        """
        Compute betweenness centrality for entities.

        Args:
            entities: Set of entity node IDs

        Returns:
            Dictionary mapping entity ID to betweenness centrality score
        """
        print("Computing betweenness centrality (this may take a while)...")
        betweenness_centrality = nx.betweenness_centrality(self.G, normalized=True)

        # Filter to entities only
        entity_betweenness = {e: betweenness_centrality[e] for e in entities}

        return entity_betweenness

    def compute_pagerank(self, entities: Set[str]) -> Dict[str, float]:
        """
        Compute PageRank for entities.

        Args:
            entities: Set of entity node IDs

        Returns:
            Dictionary mapping entity ID to PageRank score
        """
        print("Computing PageRank...")
        pagerank = nx.pagerank(self.G, alpha=0.85)

        # Filter to entities only
        entity_pagerank = {e: pagerank[e] for e in entities}

        return entity_pagerank

    def compute_document_frequency(self, entities: Set[str]) -> Dict[str, float]:
        """
        Compute document frequency (how many documents mention each entity).

        Args:
            entities: Set of entity node IDs

        Returns:
            Dictionary mapping entity ID to document frequency (count)
        """
        print("Computing document frequency...")
        doc_freq = {e: 0 for e in entities}

        # Count connections to document nodes
        for entity in entities:
            for neighbor in self.G.neighbors(entity):
                neighbor_type = self.G.nodes[neighbor].get('node_type')
                if neighbor_type == 'document':
                    doc_freq[entity] += 1

        return doc_freq

    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Min-max normalize scores to [0, 1] range.

        Args:
            scores: Dictionary of raw scores

        Returns:
            Dictionary of normalized scores
        """
        values = list(scores.values())
        if not values:
            return scores

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            # All values are the same
            return {k: 0.5 for k in scores.keys()}

        normalized = {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }

        return normalized

    def compute_horn_index(self) -> Dict[str, float]:
        """
        Compute Horn's Index for all entities.

        Returns:
            Dictionary mapping entity ID to Horn's Index score [0, 1]
        """
        entities = self.get_entities()
        print(f"Computing Horn's Index for {len(entities)} entities...")

        # Compute component metrics
        degree_scores = self.compute_degree_centrality(entities)
        betweenness_scores = self.compute_betweenness_centrality(entities)
        pagerank_scores = self.compute_pagerank(entities)
        df_scores = self.compute_document_frequency(entities)

        # Normalize all scores
        degree_norm = self.normalize_scores(degree_scores)
        betweenness_norm = self.normalize_scores(betweenness_scores)
        pagerank_norm = self.normalize_scores(pagerank_scores)
        df_norm = self.normalize_scores(df_scores)

        # Compute Horn's Index as weighted combination
        horn_index = {}
        for entity in entities:
            horn_index[entity] = (
                self.degree_weight * degree_norm[entity] +
                self.betweenness_weight * betweenness_norm[entity] +
                self.pagerank_weight * pagerank_norm[entity] +
                self.df_weight * df_norm[entity]
            )

        print("Horn's Index computation complete!")
        return horn_index

    def save_weights(self, output_path: str = 'knowledge_graph/horn_weights.json'):
        """
        Compute and save Horn's Index weights to JSON file.

        Args:
            output_path: Path to save weights JSON
        """
        horn_weights = self.compute_horn_index()

        # Sort by weight for easier inspection
        sorted_weights = dict(sorted(horn_weights.items(), key=lambda x: x[1], reverse=True))

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(sorted_weights, f, indent=2)

        print(f"Horn's Index weights saved to {output_path}")

        # Print statistics
        weights = list(horn_weights.values())
        print(f"\nStatistics:")
        print(f"  Total entities: {len(weights)}")
        if weights:
            print(f"  Mean weight: {np.mean(weights):.4f}")
            print(f"  Std weight: {np.std(weights):.4f}")
            print(f"  Min weight: {np.min(weights):.4f}")
            print(f"  Max weight: {np.max(weights):.4f}")

            # Print top-10 most important entities
            print(f"\nTop-10 Most Important Entities:")
            for i, (entity, weight) in enumerate(list(sorted_weights.items())[:10], 1):
                entity_type = self.G.nodes[entity].get('node_type', 'unknown')
                print(f"  {i}. {entity} ({entity_type}): {weight:.4f}")
        else:
            print("  No entities found!")

        return sorted_weights


def main():
    """Compute and save Horn's Index weights."""
    calculator = HornIndexCalculator()
    weights = calculator.save_weights()

    print("\n" + "=" * 80)
    print("HORN'S INDEX COMPLETE")
    print("=" * 80)
    print(f"Weights saved and ready for use in recommender system.")


if __name__ == '__main__':
    main()
