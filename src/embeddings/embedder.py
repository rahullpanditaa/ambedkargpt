import json
import numpy as np
from src.utils.constants import (
    BUFFER_MERGE_RESULTS_PATH,
    SEGMENTS_EMBEDDINGS_PATH
)
from sentence_transformers import SentenceTransformer

class MergedUnitsEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        with open(BUFFER_MERGE_RESULTS_PATH, "r") as f:
            buffer_merge_results = json.load(f)
        self.merged_units = buffer_merge_results["buffer_merge_results"]
        self.model = SentenceTransformer(model_name_or_path=model_name)
        self.segment_embeddings = None

    def _build_embeddings(self) -> np.ndarray:
        all_units_str = []
        for unit in self.merged_units:
            all_units_str.append(unit["text"])
        self.segment_embeddings = self.model.encode(all_units_str, show_progress_bar=True)
        np.save(SEGMENTS_EMBEDDINGS_PATH, self.segment_embeddings)
        return self.segment_embeddings
    
    def load_or_create_embeddings(self) -> np.ndarray:
        if SEGMENTS_EMBEDDINGS_PATH.exists():
            self.segment_embeddings = np.load(SEGMENTS_EMBEDDINGS_PATH)
            if len(self.segment_embeddings) == len(self.merged_units):
                return self.segment_embeddings
        return self._build_embeddings()

def embed_segments_command():
    em = MergedUnitsEmbedder()
    print("Generating vector embeddings for merged units (results of BufferMerge)...")
    embeddings = em.load_or_create_embeddings()
    print("Embeddings generated!!")
    print(f"Embeddings of shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")