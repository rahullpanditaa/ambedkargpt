"""
Global Graph RAG retrieval implementation (SemRAG – Equation 5).

This module implements the Global Graph RAG strategy described in the
SemRAG paper. Global search operates at the level of graph communities
instead of individual chunks or entities.

High-level idea:
1. Embed the user query
2. Retrieve the most relevant communities using community summaries
3. Use selected communities to route the query into Local Graph RAG
4. Aggregate and return locally retrieved chunks

This allows:
- High recall across the corpus
- Thematic routing before fine-grained retrieval
- Scalable retrieval over large knowledge graphs
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.retrieval.local_search import LocalGraphRAG
from src.utils.constants import (
    COMMUNITY_SUMMARIES_PATH,
    COMMUNITY_EMBEDDINGS_PATH,
    GLOBAL_SEARCH_RESULTS_PATH,
    TOP_K_COMMUNITIES,
    TOP_K_CHUNKS_PER_COMMUNITY
)


class GlobalGraphRAG:
    """
    Global Graph RAG retriever.

    This class implements Equation (5) from the SemRAG paper:
    - Communities act as high-level semantic units
    - Each community is represented by an LLM-generated summary
    - Queries are matched to communities first, then refined locally

    The retriever composes:
    - Community-level semantic search
    - Local Graph RAG (Equation 4) for final chunk scoring
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Global Graph RAG retriever.

        Loads:
        - Community summaries
        - Precomputed community embeddings
        - Initializes LocalGraphRAG for second-stage retrieval

        Args:
            model_name (str): SentenceTransformer model used for
                              query–community similarity
        """

        self.model = SentenceTransformer(model_name_or_path=model_name)

        # load community summaries
        with open(COMMUNITY_SUMMARIES_PATH, "r") as f:
            self.community_summaries: list[dict] = json.load(f)

        self.community_embeddings: np.ndarray = np.load(COMMUNITY_EMBEDDINGS_PATH)

        if len(self.community_summaries) != len(self.community_embeddings):
            raise ValueError("Mismatch between number of community summaries and number of community embeddings")

        # local Graph RAG instance (Equation 4)
        self.local_rag = LocalGraphRAG(model_name=model_name)

    def _retrieve_relevant_communities(self, user_query: str, k: int = TOP_K_COMMUNITIES) -> list[dict]:
        """
        Retrieve the most relevant communities for a user query.

        Core of Equation (5):
        similarity(query, community_summary)

        Args:
            user_query (str): User query string
            k (int): Number of top communities to retrieve

        Returns:
            list[dict]: Ranked communities with similarity scores:
                {
                    "community_id": int,
                    "score": float,
                    "summary": str
                }
        """

        # embeduser query
        query_embedding = self.model.encode([user_query])[0]

        community_scores = []

        for idx, comm_emb in enumerate(self.community_embeddings):
            score = _cosine_similarity(comm_emb, query_embedding)

            community_scores.append({
                "community_id": self.community_summaries[idx]["community_id"],
                "summary": self.community_summaries[idx]["summary"],
                "score": score
            })

        community_scores.sort(key=lambda d: d["score"], reverse=True)

        return community_scores[:k]

    def global_search(self, user_query: str, top_k_chunks: int = TOP_K_CHUNKS_PER_COMMUNITY) -> list[dict]:
        """
        Perform Global Graph RAG retrieval.

        Algorithm (SemRAG – Equation 5):
        1. Retrieve top-K relevant communities
        2. For each community:
            - Route the query into Local Graph RAG
            - Retrieve locally relevant chunks
        3. Aggregate and rank all retrieved chunks

        Args:
            user_query (str): User query string
            top_k_chunks (int): Max chunks retrieved per community

        Returns:
            list[dict]: Ranked chunks across all communities
        """

        # community-level retrieval
        top_communities = self._retrieve_relevant_communities(user_query)

        all_retrieved_chunks = []

        # send user query to local graph rag
        for comm in top_communities:
            local_chunks = self.local_rag.chunk_entity_similarity(user_query=user_query, k=top_k_chunks)

            for ch in local_chunks:
                all_retrieved_chunks.append({
                    "community_id": comm["community_id"],
                    "community_score": comm["score"],
                    "chunk_id": ch["chunk_id"],
                    "chunk_text": ch["chunk_text"],
                    "local_score": ch["score"]
                })

        # agg scores
        # final score combines:
        # - community relevance (global)
        # - chunk relevance (local)
        for item in all_retrieved_chunks:
            item["final_score"] = (
                item["community_score"] * item["local_score"]
            )

        # sort final results
        all_retrieved_chunks.sort(
            key=lambda d: d["final_score"],
            reverse=True
        )
        
        with open(GLOBAL_SEARCH_RESULTS_PATH, "w") as f:
            json.dump(
                {"global_search_results": all_retrieved_chunks},
                f,
                indent=2
            )

        return all_retrieved_chunks


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector
        vec2 (np.ndarray): Second vector

    Returns:
        float: Cosine similarity score
    """

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))
