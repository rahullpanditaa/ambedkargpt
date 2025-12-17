
# chunking - algorithm 1
# 1 - pdf ingestion - extract sentences into sentences.json / BOOK_SENTENCES_PATH
# 2 - buffer merge sentences.json - src/chunking/buffer_merger.py
# 3 - buffer merge results processing - compute merged units embedding - cosine distance bw adjacent units - chunking/..embedder.py
# 4 - Algorithm 1 implementation - Semantic chunking  - src/chunking/semantic_chunker.py

# Graph:
# 5 - entity extraction - semantic chunks - entity extraction per chunk - src/graph/entity_extractor.py
# 6 - relationship extractor - derive relationship bw entities based on co occurrence - src/graph/relationship_Extractor.py
# 7 - graph build - build graph from entity relations extracted - src/graph/graph_builder.py
# 8 - community detection - run leiden algorithm for community detection on knowledge graph - src/graph/community_detector.py
# 9 - summarizer - generate summaries for each community - src/graph/summarizer.py