"""Pytest configuration for the tests folder.

Two old scripts are kept out of test collection until they are repaired in a
refactor/ branch. Both stop pytest during collection (exit code 2), so no test
would run at all:

- test_concept_similarity.py runs a WordNet analysis as soon as it is imported
  and needs NLTK corpora that are not installed in CI.
- test_production_infrastructure.py imports HighSeverityError, which no longer
  exists in utils/error_handler.py.
"""

collect_ignore = ["test_concept_similarity.py", "test_production_infrastructure.py"]
