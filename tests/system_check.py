"""
System Health Check
Run this before executing any scripts to verify all dependencies and files
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path


def check_file(filepath, description, required=True):
    """Check if a file exists and report status."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else ("❌ MISSING" if required else "⚠️ OPTIONAL")

    size = ""
    if exists:
        file_size = os.path.getsize(filepath)
        if file_size < 1024:
            size = f"({file_size} B)"
        elif file_size < 1024**2:
            size = f"({file_size / 1024:.1f} KB)"
        else:
            size = f"({file_size / (1024**2):.1f} MB)"

    print(f"{status:15} {description:40} {filepath} {size}")
    return exists


def check_json_encoding(filepath):
    """Check if JSON file can be loaded with UTF-8."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            json.load(f)
        return True, "Valid JSON"
    except UnicodeDecodeError as e:
        return False, f"Encoding error: {e}"
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    print("=" * 80)
    print("HERITAGE RECOMMENDER SYSTEM - DIAGNOSTICS")
    print("=" * 80)

    print("\n📂 FILE SYSTEM CHECK")
    print("-" * 80)

    # Critical files
    critical_files = [
        ("data/classified/classified_documents.json", "Classified Documents", True),
        ("data/embeddings/document_embeddings.npy", "Document Embeddings", True),
        ("data/embeddings/embedding_mapping.json", "Embedding Metadata", True),
        ("knowledge_graph/heritage_kg.gpickle", "Knowledge Graph", True),
        ("knowledge_graph/simrank/simrank_matrix.npy", "SimRank Matrix", True),
        ("models/ranker/faiss/flat_index.faiss", "FAISS Index (Flat)", True),
        ("models/ranker/faiss/hnsw_index.faiss", "FAISS Index (HNSW)", False),
    ]

    missing_critical = []
    for filepath, desc, required in critical_files:
        exists = check_file(filepath, desc, required)
        if required and not exists:
            missing_critical.append((filepath, desc))

    # Optional files
    print("\n📋 OPTIONAL FILES")
    print("-" * 80)

    optional_files = [
        ("knowledge_graph/horn_weights.json", "Horn's Index Weights", False),
        ("data/evaluation/ground_truth.json", "Ground Truth (Old)", False),
        ("data/evaluation/ground_truth_improved.json", "Ground Truth (Improved)", False),
        ("results/evaluation_results.json", "Evaluation Results", False),
    ]

    for filepath, desc, required in optional_files:
        check_file(filepath, desc, required)

    # JSON encoding check
    print("\n🔍 JSON ENCODING CHECK")
    print("-" * 80)

    json_files = [
        "data/classified/classified_documents.json",
        "data/embeddings/embedding_mapping.json",
    ]

    for json_file in json_files:
        if os.path.exists(json_file):
            valid, msg = check_json_encoding(json_file)
            status = "✅" if valid else "❌"
            print(f"{status} {json_file}: {msg}")

    # Data shape check
    print("\n📊 DATA SHAPE VERIFICATION")
    print("-" * 80)

    try:
        # Load and check documents
        with open("data/classified/classified_documents.json", "r", encoding="utf-8") as f:
            documents = json.load(f)
        print(f"✅ Documents: {len(documents)} items")

        # Check first document structure
        if documents:
            doc = documents[0]
            required_keys = ["title", "cluster_id", "classifications", "entities"]
            missing_keys = [k for k in required_keys if k not in doc]
            if missing_keys:
                print(f"   ⚠️ Missing keys in documents: {missing_keys}")
            else:
                print(f"   ✅ Document structure valid")
    except Exception as e:
        print(f"❌ Documents: {e}")

    try:
        # Load and check embeddings
        embeddings = np.load("data/embeddings/document_embeddings.npy")
        print(f"✅ Embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}")

        if len(documents) != embeddings.shape[0]:
            print(f"   ⚠️ WARNING: Document count ({len(documents)}) != Embedding count ({embeddings.shape[0]})")
    except Exception as e:
        print(f"❌ Embeddings: {e}")

    try:
        # Load and check SimRank
        simrank = np.load("knowledge_graph/simrank/simrank_matrix.npy")
        print(f"✅ SimRank: shape {simrank.shape}, dtype {simrank.dtype}")

        if simrank.shape[0] != len(documents):
            print(f"   ⚠️ WARNING: SimRank size ({simrank.shape[0]}) != Document count ({len(documents)})")
    except Exception as e:
        print(f"❌ SimRank: {e}")

    try:
        # Load and check KG
        with open("knowledge_graph/heritage_kg.gpickle", "rb") as f:
            G = pickle.load(f)
        print(f"✅ Knowledge Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Count document nodes
        doc_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "document"]
        print(f"   Document nodes in KG: {len(doc_nodes)}")

        if len(doc_nodes) != len(documents):
            print(f"   ⚠️ WARNING: KG document nodes ({len(doc_nodes)}) != Document count ({len(documents)})")
    except Exception as e:
        print(f"❌ Knowledge Graph: {e}")

    # Python packages check
    print("\n📦 PYTHON PACKAGES")
    print("-" * 80)

    required_packages = [
        "numpy",
        "pandas",
        "networkx",
        "nltk",
        "spacy",
        "sentence_transformers",
        "faiss",
        "torch",
        "sklearn",
        "streamlit",
        "plotly",
        "pyvis",
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package:25} installed")
        except ImportError:
            print(f"❌ {package:25} MISSING - install with: pip install {package}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if missing_critical:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for filepath, desc in missing_critical:
            print(f"   - Missing: {desc} ({filepath})")
        print("\n⚠️ Cannot proceed until critical files are available.")
        print("   Run the data collection and preprocessing pipeline first.")
    else:
        print("\n✅ ALL CRITICAL FILES PRESENT")

        if not os.path.exists("knowledge_graph/horn_weights.json"):
            print("\n⚠️ RECOMMENDED: Generate Horn's Index weights")
            print("   Run: python src/4_knowledge_graph/horn_index.py")

        if not os.path.exists("data/evaluation/ground_truth_improved.json"):
            print("\n⚠️ RECOMMENDED: Generate improved ground truth")
            print("   Run: python src/7_evaluation/create_ground_truth.py")

        print("\n🚀 System is ready to use!")

    print("=" * 80)


if __name__ == "__main__":
    main()
