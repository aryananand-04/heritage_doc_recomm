"""
Query Processor for Heritage Document Recommendation System

Parses natural language queries to extract:
- Heritage types (temple, fort, monument, etc.)
- Domains (religious, military, royal, etc.)
- Time periods (ancient, medieval, modern)
- Regions (North India, Rajasthan, etc.)
- Architectural styles (Mughal, Dravidian, etc.)
- Named entities (locations, persons, organizations)

Uses spaCy for NLP and classification schemas from metadata extraction.
"""

import spacy
import re
from typing import Dict, List, Set
from sentence_transformers import SentenceTransformer
import numpy as np


class QueryProcessor:
    """Processes natural language queries for heritage document search."""

    # Classification schemas (from 2_extract_metadata_spaCy.py)
    HERITAGE_TYPES = {
        'monument', 'site', 'artifact', 'architecture', 'tradition', 'art',
        'temple', 'fort', 'palace', 'mosque', 'church', 'monastery', 'stupa',
        'pagoda', 'shrine', 'cathedral', 'fortress', 'citadel', 'castle',
        'mansion', 'haveli', 'memorial', 'statue', 'tomb', 'cenotaph'
    }

    DOMAINS = {
        'religious', 'military', 'royal', 'cultural', 'archaeological', 'architectural',
        'spiritual', 'defensive', 'residential', 'commemorative', 'sacred', 'worship'
    }

    TIME_PERIODS = {
        'ancient': ['ancient', 'prehistoric', 'vedic', 'maurya', 'gupta'],
        'medieval': ['medieval', 'mughal', 'sultanate', 'vijayanagara', 'maratha', 'rajput'],
        'modern': ['modern', 'colonial', 'british', 'contemporary', 'post-independence']
    }

    INDIAN_REGIONS = {
        'north': ['north', 'northern', 'delhi', 'punjab', 'haryana', 'himachal', 'uttarakhand', 'uttar pradesh', 'jammu', 'kashmir'],
        'south': ['south', 'southern', 'tamil nadu', 'kerala', 'karnataka', 'andhra pradesh', 'telangana'],
        'east': ['east', 'eastern', 'west bengal', 'odisha', 'bihar', 'jharkhand', 'sikkim'],
        'west': ['west', 'western', 'rajasthan', 'gujarat', 'maharashtra', 'goa'],
        'central': ['central', 'madhya pradesh', 'chhattisgarh']
    }

    ARCHITECTURAL_STYLES = {
        'indo-islamic', 'mughal', 'dravidian', 'nagara', 'vesara', 'buddhist',
        'colonial', 'rajput', 'maratha', 'vijayanagara', 'hoysala', 'chalukya',
        'pallava', 'chola', 'indo-saracenic'
    }

    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize query processor.

        Args:
            embedding_model_name: Name of sentence transformer model
        """
        print(f"Loading spaCy model...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')

        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Query processor initialized successfully!")

    def parse_query(self, query_text: str) -> Dict:
        """
        Parse natural language query to extract heritage attributes.

        Args:
            query_text: Natural language query (e.g., "Mughal temples in North India")

        Returns:
            Dictionary with extracted attributes:
            - heritage_types: Set of heritage types
            - domains: Set of domains
            - time_period: Detected time period (ancient/medieval/modern)
            - region: Detected region (north/south/east/west/central)
            - architectural_styles: Set of architectural styles
            - locations: List of location entities
            - persons: List of person entities
            - organizations: List of organization entities
            - query_embedding: 384-dim embedding vector
            - original_query: Original query text
        """
        query_lower = query_text.lower()
        doc = self.nlp(query_text)

        parsed = {
            'heritage_types': set(),
            'domains': set(),
            'time_period': None,
            'region': None,
            'architectural_styles': set(),
            'locations': [],
            'persons': [],
            'organizations': [],
            'query_embedding': None,
            'original_query': query_text
        }

        # Extract heritage types
        for heritage_type in self.HERITAGE_TYPES:
            if heritage_type in query_lower:
                parsed['heritage_types'].add(heritage_type)

        # Extract domains
        for domain in self.DOMAINS:
            if domain in query_lower:
                parsed['domains'].add(domain)

        # Extract time period
        for period, keywords in self.TIME_PERIODS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    parsed['time_period'] = period
                    break
            if parsed['time_period']:
                break

        # Extract region
        for region, keywords in self.INDIAN_REGIONS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    parsed['region'] = region
                    break
            if parsed['region']:
                break

        # Extract architectural styles
        for style in self.ARCHITECTURAL_STYLES:
            if style in query_lower:
                parsed['architectural_styles'].add(style)

        # Extract named entities using spaCy
        for ent in doc.ents:
            if ent.label_ == 'GPE' or ent.label_ == 'LOC':
                parsed['locations'].append(ent.text)
            elif ent.label_ == 'PERSON':
                parsed['persons'].append(ent.text)
            elif ent.label_ == 'ORG':
                parsed['organizations'].append(ent.text)

        # Generate query embedding
        parsed['query_embedding'] = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True
        )

        return parsed

    def format_parsed_query(self, parsed: Dict) -> str:
        """
        Format parsed query for display.

        Args:
            parsed: Parsed query dictionary

        Returns:
            Formatted string representation
        """
        lines = [
            f"Query: {parsed['original_query']}",
            "-" * 60
        ]

        if parsed['heritage_types']:
            lines.append(f"Heritage Types: {', '.join(parsed['heritage_types'])}")

        if parsed['domains']:
            lines.append(f"Domains: {', '.join(parsed['domains'])}")

        if parsed['time_period']:
            lines.append(f"Time Period: {parsed['time_period']}")

        if parsed['region']:
            lines.append(f"Region: {parsed['region']}")

        if parsed['architectural_styles']:
            lines.append(f"Architectural Styles: {', '.join(parsed['architectural_styles'])}")

        if parsed['locations']:
            lines.append(f"Locations: {', '.join(parsed['locations'])}")

        if parsed['persons']:
            lines.append(f"Persons: {', '.join(parsed['persons'])}")

        if parsed['organizations']:
            lines.append(f"Organizations: {', '.join(parsed['organizations'])}")

        lines.append(f"Embedding: {parsed['query_embedding'].shape}")

        return '\n'.join(lines)


def main():
    """Test query processor with example queries."""
    processor = QueryProcessor()

    test_queries = [
        "Mughal temples in North India",
        "Ancient forts in Rajasthan",
        "Buddhist stupas and monasteries",
        "Dravidian temples in South India",
        "Colonial architecture in Mumbai",
        "Medieval palaces of Maratha kings",
        "Religious monuments of Vijayanagara empire"
    ]

    print("\n" + "=" * 80)
    print("QUERY PROCESSOR TEST")
    print("=" * 80 + "\n")

    for query in test_queries:
        parsed = processor.parse_query(query)
        print(processor.format_parsed_query(parsed))
        print()


if __name__ == '__main__':
    main()
