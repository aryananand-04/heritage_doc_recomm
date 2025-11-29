"""Test the complete query system end-to-end."""

import sys
sys.path.append('src/6_query_system')

from query_processor import QueryProcessor
from recommender import HeritageRecommender

def main():
    print("=" * 80)
    print("HERITAGE DOCUMENT RECOMMENDATION SYSTEM - END-TO-END TEST")
    print("=" * 80)

    # Initialize components
    print("\n[1/2] Initializing Query Processor...")
    processor = QueryProcessor()

    print("\n[2/2] Initializing Recommender...")
    recommender = HeritageRecommender()

    # Test queries
    test_queries = [
        "Mughal temples in North India",
        "Ancient forts in Rajasthan",
        "Buddhist stupas and monasteries"
    ]

    print("\n" + "=" * 80)
    print("RUNNING TEST QUERIES")
    print("=" * 80)

    for i, query_text in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"QUERY {i}: {query_text}")
        print("=" * 80)

        # Parse query
        print("\n[Parsing Query...]")
        parsed_query = processor.parse_query(query_text)
        print(processor.format_parsed_query(parsed_query))

        # Get recommendations
        print("\n[Getting Recommendations...]")
        recommendations = recommender.recommend(parsed_query, top_k=5, explain=True)

        # Display results
        print(f"\nTop-5 Recommendations:")
        print("-" * 80)
        for rec in recommendations:
            print(recommender.format_recommendation(rec))
            print()

    print("=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
