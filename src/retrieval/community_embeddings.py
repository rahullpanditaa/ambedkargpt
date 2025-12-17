import json
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
COMMUNITY_SUMMARIES_PATH = PROCESSED_DATA_DIR_PATH / "community_summaries.json"
COMMUNITY_EMBEDDINGS_PATH = PROCESSED_DATA_DIR_PATH / "community_embeddings.npy"

class GenerateCommunityEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(COMMUNITY_SUMMARIES_PATH, "r") as f:
            self.community_summaries: list[dict] = json.load(f)
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def embed_summaries(self) -> np.ndarray:
        summary_texts = []
        for comm_summary  in self.community_summaries:
            summary_text = comm_summary["summary"]
            summary_texts.append(summary_text)
        embeddings = self.model.encode(summary_texts)

        np.save(COMMUNITY_EMBEDDINGS_PATH, embeddings)
        return embeddings