import pickle
import json
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter

class HornIndexCalculator:
    """Calculate Horn's Index for entity importance in KG."""
    
    def __init__(
        self,
        kg_path='knowledge_graph/heritage_kg.gpickle',
        degree_weight=0.3,
        betweenness_weight=0.2,
        pagerank_weight=0.3,
        df_weight=0.2
    ):
        print(f"📂 Loading KG from {kg_path}...")
        with open(kg_path, 'rb') as f:
            self.G = pickle.load(f)
        
        self.weights = {
            'degree': degree_weight,
            'betweenness': betweenness_weight,
            'pagerank': pagerank_weight,
            'df': df_weight
        }
        
        self.entity_types = {
            'location', 'person', 'organization', 
            'heritage_type', 'domain', 'time_period', 'region'
        }
        
        print(f"✓ Loaded KG: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    
    def get_entities(self):
        """Extract all entity nodes."""
        entities = set()
        for node, data in self.G.nodes(data=True):
            if data.get('node_type') in self.entity_types:
                entities.add(node)
        return entities
    
    def compute_degree_centrality(self, entities):
        """Degree centrality for entities."""
        print("📊 Computing degree centrality...")
        degree_cent = nx.degree_centrality(self.G)
        return {e: degree_cent[e] for e in entities}
    
    def compute_betweenness_centrality(self, entities):
        """Betweenness centrality (may take time)."""
        print("🔗 Computing betweenness centrality (this may take 2-3 minutes)...")
        try:
            # Sample 100 nodes for approximation (faster)
            between_cent = nx.betweenness_centrality(self.G, normalized=True, k=min(100, self.G.number_of_nodes()))
            return {e: between_cent.get(e, 0.0) for e in entities}
        except Exception as e:
            print(f"   ⚠️ Betweenness computation failed: {e}")
            print("   Using zeros for betweenness...")
            return {e: 0.0 for e in entities}
    
    def compute_pagerank(self, entities):
        """PageRank for entities."""
        print("🔍 Computing PageRank...")
        pagerank = nx.pagerank(self.G, alpha=0.85, max_iter=100)
        return {e: pagerank[e] for e in entities}
    
    def compute_document_frequency(self, entities):
        """Count document connections for each entity."""
        print("📄 Computing document frequency...")
        df = {e: 0 for e in entities}
        
        for entity in entities:
            for neighbor in self.G.neighbors(entity):
                if self.G.nodes[neighbor].get('node_type') == 'document':
                    df[entity] += 1
        
        return df
    
    def normalize(self, scores):
        """Min-max normalization to [0, 1]."""
        values = list(scores.values())
        if not values:
            return scores
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 0.5 for k in scores}
        
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
    
    def compute_horn_index(self):
        """Compute Horn's Index for all entities."""
        entities = self.get_entities()
        print(f"\n🎯 Computing Horn's Index for {len(entities)} entities...\n")
        
        # Compute components
        degree = self.compute_degree_centrality(entities)
        betweenness = self.compute_betweenness_centrality(entities)
        pagerank = self.compute_pagerank(entities)
        df = self.compute_document_frequency(entities)
        
        # Normalize
        print("\n📐 Normalizing scores...")
        degree_norm = self.normalize(degree)
        between_norm = self.normalize(betweenness)
        pagerank_norm = self.normalize(pagerank)
        df_norm = self.normalize(df)
        
        # Weighted combination
        print("⚖️  Computing weighted Horn's Index...")
        horn_index = {}
        for entity in entities:
            horn_index[entity] = (
                self.weights['degree'] * degree_norm[entity] +
                self.weights['betweenness'] * between_norm[entity] +
                self.weights['pagerank'] * pagerank_norm[entity] +
                self.weights['df'] * df_norm[entity]
            )
        
        print("✅ Horn's Index computation complete!\n")
        return horn_index
    
    def save_weights(self, output_path='knowledge_graph/horn_weights.json'):
        """Compute and save Horn's Index weights."""
        horn_weights = self.compute_horn_index()
        
        # Sort by importance
        sorted_weights = dict(sorted(horn_weights.items(), key=lambda x: x[1], reverse=True))
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sorted_weights, f, indent=2)
        
        print(f"💾 Saved to: {output_path}")
        
        # Statistics
        weights = list(horn_weights.values())
        print(f"\n📊 Statistics:")
        print(f"   Total entities: {len(weights)}")
        print(f"   Mean weight: {np.mean(weights):.4f}")
        print(f"   Std weight: {np.std(weights):.4f}")
        print(f"   Min weight: {np.min(weights):.4f}")
        print(f"   Max weight: {np.max(weights):.4f}")
        
        # Top entities
        print(f"\n🏆 Top-10 Most Important Entities:")
        for i, (entity, weight) in enumerate(list(sorted_weights.items())[:10], 1):
            entity_type = self.G.nodes[entity].get('node_type', 'unknown')
            entity_name = self.G.nodes[entity].get('name', entity)
            print(f"   {i}. {entity_name} ({entity_type}): {weight:.4f}")
        
        return sorted_weights


def main():
    print("="*80)
    print("HORN'S INDEX CALCULATOR")
    print("="*80 + "\n")
    
    calculator = HornIndexCalculator()
    weights = calculator.save_weights()
    
    print("\n" + "="*80)
    print("✅ COMPLETE - Horn's Index weights ready for recommender system")
    print("="*80)


if __name__ == '__main__':
    main()