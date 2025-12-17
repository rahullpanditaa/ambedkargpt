"""
Community embedding generation for Graph RAG retrieval.

This module embeds community-level summaries into dense vectors.
These embeddings are used during retrieval to score and rank
communities against a query, particularly in Global Graph RAG
(Equation 5 in the SemRAG paper).
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.constants import (
    PROCESSED_DATA_DIR_PATH, COMMUNITY_SUMMARIES_PATH, COMMUNITY_EMBEDDINGS_PATH
)

class GenerateCommunityEmbeddings:
    """
    Generates vector embeddings for community summaries.

    This class:
    - Loads LLM-generated community summaries
    - Encodes each summary using a sentence-transformer model
    - Persists the resulting embeddings to disk

    The embeddings serve as high-level semantic representations
    of graph communities for retrieval and routing.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(COMMUNITY_SUMMARIES_PATH, "r") as f:
            self.community_summaries: list[dict] = json.load(f)
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def embed_summaries(self) -> np.ndarray:
        """
        Embed all community summaries into dense vectors.

        Returns:
            np.ndarray: Array of shape (num_communities, embedding_dim)

        Side Effects:
            - Saves embeddings to COMMUNITY_EMBEDDINGS_PATH
        """
        summary_texts = []
        for comm_summary  in self.community_summaries:
            summary_text = comm_summary["summary"]
            summary_texts.append(summary_text)
        embeddings = self.model.encode(summary_texts)

        np.save(COMMUNITY_EMBEDDINGS_PATH, embeddings)
        return embeddings