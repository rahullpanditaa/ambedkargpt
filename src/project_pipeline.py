
# 1 - pdf ingestion - extract sentences into sentences.json / BOOK_SENTENCES_PATH
# 2 - buffer merge sentences.json - src/chunking/buffer_merger.py
# 3 - buffer merge results processing - compute merged units embedding - cosine distance bw adjacent units - chunking/..embedder.py