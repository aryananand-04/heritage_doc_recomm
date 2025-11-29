import json
from nltk.corpus import wordnet as wn

def lesk_similarity(word1, word2):
    """Current implementation from KG builder"""
    try:
        word1_clean = word1.lower().replace(' ', '_').replace('-', '_')
        word2_clean = word2.lower().replace(' ', '_').replace('-', '_')

        synsets1 = wn.synsets(word1_clean)
        synsets2 = wn.synsets(word2_clean)

        if not synsets1 or not synsets2:
            return 0.0

        max_sim = 0.0
        for s1 in synsets1[:5]:
            for s2 in synsets2[:5]:
                try:
                    sim = s1.path_similarity(s2)
                    if sim and sim > max_sim:
                        max_sim = sim

                    if not sim or sim < 0.3:
                        wup_sim = s1.wup_similarity(s2)
                        if wup_sim and wup_sim > max_sim:
                            max_sim = wup_sim
                except:
                    continue

        return max_sim
    except Exception as e:
        return 0.0

def compute_concept_similarity(concept1, concept2):
    """Current implementation from KG builder"""
    if concept1.lower() == concept2.lower():
        return 1.0

    related_groups = [
        {'temple', 'mosque', 'church', 'monastery', 'shrine', 'cathedral', 'stupa', 'pagoda'},
        {'fort', 'fortress', 'citadel', 'castle', 'stronghold', 'garrison'},
        {'palace', 'mansion', 'haveli', 'royal_residence'},
        {'ancient', 'medieval', 'modern', 'prehistoric', 'contemporary'},
        {'monument', 'memorial', 'statue', 'cenotaph', 'tomb'},
    ]

    for group in related_groups:
        c1_lower = concept1.lower().replace(' ', '_')
        c2_lower = concept2.lower().replace(' ', '_')

        if c1_lower in group and c2_lower in group:
            return 0.8

    lesk_sim = lesk_similarity(concept1, concept2)
    return lesk_sim

# Test cases
print("=" * 80)
print("CONCEPT SIMILARITY TESTING")
print("=" * 80)

test_pairs = [
    ('temple', 'mosque'),
    ('temple', 'church'),
    ('mosque', 'church'),
    ('fort', 'fortress'),
    ('palace', 'mansion'),
    ('monument', 'memorial'),
    ('ancient', 'medieval'),
    ('temple', 'fort'),  # Should be low
    ('religious', 'military'),  # Should be low
]

print(f"\n{'Concept 1':<20} {'Concept 2':<20} {'Lesk':<10} {'Domain Group':<15} {'Final':<10} {'Status':<10}")
print("-" * 80)

threshold = 0.5

for c1, c2 in test_pairs:
    lesk_sim = lesk_similarity(c1, c2)
    final_sim = compute_concept_similarity(c1, c2)

    # Check if in same group
    in_group = "Yes" if final_sim == 0.8 else "No"
    passes = "✓" if final_sim > threshold else "✗"

    print(f"{c1:<20} {c2:<20} {lesk_sim:<10.3f} {in_group:<15} {final_sim:<10.3f} {passes:<10}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# Check what's in the actual KG
with open('knowledge_graph/kg_statistics.json', 'r') as f:
    stats = json.load(f)

print(f"\nConcept similarity edges found: {stats['edge_types'].get('semantically_related', 0)}")
print(f"Threshold used: {stats['config']['concept_similarity_threshold']}")

# Check WordNet synsets
print("\n" + "=" * 80)
print("WORDNET ANALYSIS")
print("=" * 80)

for word in ['temple', 'mosque', 'church']:
    synsets = wn.synsets(word)
    print(f"\n{word.upper()}:")
    for i, syn in enumerate(synsets[:3]):
        print(f"  {i+1}. {syn.name()}: {syn.definition()}")

# Test direct path similarity
print("\n" + "=" * 80)
print("DIRECT WORDNET SIMILARITY")
print("=" * 80)

pairs_to_test = [('temple', 'mosque'), ('temple', 'church'), ('mosque', 'church')]
for w1, w2 in pairs_to_test:
    s1 = wn.synsets(w1)
    s2 = wn.synsets(w2)

    if s1 and s2:
        best_path = max((s1[i].path_similarity(s2[j]) or 0)
                       for i in range(min(3, len(s1)))
                       for j in range(min(3, len(s2))))
        best_wup = max((s1[i].wup_similarity(s2[j]) or 0)
                      for i in range(min(3, len(s1)))
                      for j in range(min(3, len(s2))))

        print(f"{w1} <-> {w2}:")
        print(f"  Path similarity: {best_path:.3f}")
        print(f"  WUP similarity:  {best_wup:.3f}")
