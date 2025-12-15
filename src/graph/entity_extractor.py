import json
from pathlib import Path
import spacy

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
CHUNKS_OUTPUT_PATH = PROCESSED_DATA_DIR_PATH / "chunks.json"
CHUNK_ENTITIES_PATH = PROCESSED_DATA_DIR_PATH / "chunk_entities.json"

nlp = spacy.load("en_core_web_sm")

class EntityExtractor:
    def __init__(self):
        with open(CHUNKS_OUTPUT_PATH, "r") as f:
            chunks = json.load(f)
        # keys - chunk id, text, ...
        self.chunks: list[dict] = chunks["chunks"]
        self.model = spacy.load("en_core_web_sm")

    def extract_entities(self) -> list[dict]:
        entities = []
        for chunk in self.chunks:
            doc = nlp(chunk["text"])
            current_chunk_entities = []
            for ent in doc.ents:
                current_chunk_entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
            entities.append({
                "chunk_id": chunk["chunk_id"],
                "entities": current_chunk_entities
            })
        return entities
    
    def load_or_create_chunk_entities(self):
        entities = self.extract_entities()
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(CHUNK_ENTITIES_PATH, "w") as f:
            json.dump({"chunk_entities": entities}, f, indent=2)
        