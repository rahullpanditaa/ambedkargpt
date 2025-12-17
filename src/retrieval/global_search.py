import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DATA_DIR_PATH = Path(__file__).parent.parent.parent / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"

COMMUNITY_SUMMARIES_PATH = PROCESSED_DATA_DIR_PATH / "community_summaries.json"
COMMUNITY_EMBEDDINGS_PATH = PROCESSED_DATA_DIR_PATH / "community_embeddings.npy"
COMMUNITY_CHUNKS_PATH = PROCESSED_DATA_DIR_PATH / "community_chunks.json"
CHUNKS_PATH = PROCESSED_DATA_DIR_PATH / "chunks.json"


class GlobalGraphRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name_or_path=model_name)

        # load summaries
        with open(COMMUNITY_SUMMARIES_PATH, "r") as f:
            self.community_summaries = json.load(f)

        # load embeddings
        self.community_embeddings = np.load(COMMUNITY_EMBEDDINGS_PATH)

        # load community -> chunk_ids
        with open(COMMUNITY_CHUNKS_PATH, "r") as f:
            self.community_to_chunks = json.load(f)

        # load chunks
        with open(CHUNKS_PATH, "r") as f:
            self.chunks = json.load(f)["chunks"]

        self.chunk_id_to_text = {
            ch["chunk_id"]: ch["text"] for ch in self.chunks
        }

    def global_search(self, query: str, top_k_communities=3, top_k_chunks=8):
        query_emb = self.model.encode([query], normalize_embeddings=True)

        # similarity with community summaries
        sim_scores = cosine_similarity(query_emb, self.community_embeddings)[0]

        community_scores = []
        for idx, score in enumerate(sim_scores):
            community_id = self.community_summaries[idx]["community_id"]
            community_scores.append({
                "community_id": community_id,
                "score": float(score)
            })

        # select top k communities
        community_scores.sort(key=lambda d: d["score"], reverse=True)
        top_communities = community_scores[:top_k_communities]

        # 
        candidate_chunk_ids = set()
        for c in top_communities:
            cid = str(c["community_id"])
            candidate_chunk_ids.update(self.community_to_chunks.get(cid, []))

        # Step 5: Score chunks w.r.t query
        candidate_texts = []
        chunk_ids = []

        for cid in candidate_chunk_ids:
            if cid in self.chunk_id_to_text:
                candidate_texts.append(self.chunk_id_to_text[cid])
                chunk_ids.append(cid)

        if not candidate_texts:
            return []

        chunk_embeddings = self.embedding_model.encode(
            candidate_texts, normalize_embeddings=True
        )

        chunk_scores = cosine_similarity(query_emb, chunk_embeddings)[0]

        scored_chunks = []
        for i, score in enumerate(chunk_scores):
            scored_chunks.append({
                "chunk_id": chunk_ids[i],
                "chunk_text": candidate_texts[i],
                "score": float(score)
            })

        # Step 6: Rank and select top chunks
        scored_chunks.sort(key=lambda d: d["score"], reverse=True)

        return scored_chunks[:top_k_chunks]
