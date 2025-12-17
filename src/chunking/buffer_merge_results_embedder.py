"""
Embedding and cosine-distance computation for buffer-merged units.

This module implements the embedding and distance computation steps
of Algorithm 1 in the SemRAG paper:

- Step 4: Embed buffer-merged units (Ŝ)
- Steps 5–6: Compute cosine distance between adjacent embeddings

The resulting artifacts are:
- Segment embeddings (numpy array)
- Cosine distances between consecutive segments
"""

import json
import numpy as np
from src.utils.constants import (
    BUFFER_MERGE_RESULTS_PATH,
    SEGMENTS_EMBEDDINGS_PATH,
    SEGMENTS_DISTANCES_PATH
)
from sentence_transformers import SentenceTransformer

class MergedUnitsEmbedder:
    """
    Handles embedding of buffer-merged units and computation of
    cosine distances between adjacent units.

    This class consumes:
    - Buffer-merged units (Ŝ)

    And produces:
    - Vector embeddings for each merged unit
    - Cosine distance array aligned with merged-unit ordering

    These outputs are used directly by the semantic chunking stage.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedder.

        Args:
            model_name (str): SentenceTransformer model used to embed
                              merged units.

        Side Effects:
            - Loads buffer-merged units from disk
            - Initializes the embedding model
        """
        with open(BUFFER_MERGE_RESULTS_PATH, "r") as f:
            buffer_merge_results = json.load(f)
        self.merged_units = buffer_merge_results["buffer_merge_results"]
        self.model = SentenceTransformer(model_name_or_path=model_name)
        # self.segment_embeddings = None

    def _build_embeddings(self) -> np.ndarray:
        """
        Generate embeddings for all buffer-merged units.

        Returns:
            np.ndarray: Embedding matrix of shape (N, D),
                        where N is the number of merged units
                        and D is the embedding dimension.

        Side Effects:
            - Saves embeddings to SEGMENTS_EMBEDDINGS_PATH
        """
        all_units_str = []
        for unit in self.merged_units:
            all_units_str.append(unit["text"])
        segment_embeddings = self.model.encode(all_units_str, show_progress_bar=True)
        np.save(SEGMENTS_EMBEDDINGS_PATH, segment_embeddings)
        return segment_embeddings
    
    def _load_or_create_embeddings(self) -> np.ndarray:
        """
        Load precomputed embeddings if available, otherwise generate them.

        Ensures that the number of loaded embeddings matches the number
        of merged units before accepting cached results.

        Returns:
            np.ndarray: Segment embeddings aligned with merged-unit order
        """
        if SEGMENTS_EMBEDDINGS_PATH.exists():
            segment_embeddings = np.load(SEGMENTS_EMBEDDINGS_PATH)
            if len(segment_embeddings) == len(self.merged_units):
                return segment_embeddings
        # Algorithm 1 step 4
        return self._build_embeddings()
    
    def compute_cosine_distance(self):
        """
        Compute cosine distance between consecutive segment embeddings.

        For embeddings e[i] and e[i+1], the distance is defined as:
            d = 1 - cosine_similarity(e[i], e[i+1])

        This produces an array of length N-1, aligned such that:
            distances[i] = distance between segment i and i+1
        """
        # if self.segment_embeddings is None or len(self.segment_embeddings) == 0:
        #     raise ValueError("No segment embeddings loaded. Call 'load_or_create_embeddings'")
        segment_embeddings = self._load_or_create_embeddings()
        if SEGMENTS_DISTANCES_PATH.exists():
            print(f"Cosine distances already calculated. File at '{SEGMENTS_DISTANCES_PATH.name}'")
            return
        
        cos_distances = []
        for i in range(len(segment_embeddings) - 1):
            d = 1 - _cosine_similarity(segment_embeddings[i],
                                       segment_embeddings[i+1])
            # Algorithm 1 steps 5,6
            cos_distances.append(d)

        numpy_arr = np.array(cos_distances, dtype=np.float32)
        np.save(SEGMENTS_DISTANCES_PATH, numpy_arr)

    def _distances_inspection(self):
        """
        Print basic statistics for computed cosine distances.

        Intended for manual inspection and threshold tuning.
        """
        distances = np.load(SEGMENTS_DISTANCES_PATH)
        print("Inspecting calculated cosine distances:")
        print(f"- Mean: {np.mean(distances):.4f}")
        print(f"- Standard deviation: {np.std(distances):.4f}")
        print(f"- Maximum distance: {np.amax(distances):.4f}")
        print(f"- Minimum distance: {np.amin(distances):.4f}")
        

# def embed_segments_command():
#     em = MergedUnitsEmbedder()
#     print("Generating vector embeddings for merged units (results of BufferMerge)...")
#     embeddings = em.load_or_create_embeddings()
#     print("Embeddings generated!!")
#     print(f"Embeddings of shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

# def compute_cosine_distances_command():
#     em = MergedUnitsEmbedder()
#     print("Computing cosine distances between consecutive segment embeddings...")
#     em.load_or_create_embeddings()
#     em.compute_cosine_distance()
#     print(f"- Distances computed. Saved to 'data/processed/{SEGMENTS_DISTANCES_PATH.name}'")

# def inspect_distances_command():
#     em = MergedUnitsEmbedder()
#     em.load_or_create_embeddings()
#     em.distances_inspection()

def _cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)