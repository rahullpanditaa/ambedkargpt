import json
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
COMMUNITY_SUMMARIES_PATH = PROCESSED_DATA_DIR_PATH / "community_summaries.json"
COMMUNITY_EMBEDDINGS_PATH = PROCESSED_DATA_DIR_PATH / "community_embeddings.npy"
CHUNK_ENTITIES_PATH = PROCESSED_DATA_DIR_PATH / "chunk_entities.json"

class LocalGraphRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def _retrieve_entitity(self, user_query: str):
        query_embedding = self.model.encode([user_query])[0]

        
        # need embedding for every entity
        with open(CHUNK_ENTITIES_PATH, "r") as f:
            chunk_entities: list[dict] = json.load(f)["chunk_entities"]
        entity_texts = []
        for chunk in chunk_entities:
            entities = chunk["entities"]
            for ent in entities:
                entity_texts.append(ent["text_norm"])
    