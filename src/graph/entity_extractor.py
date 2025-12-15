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

    def extract_entities(self) -> list[dict]:
        entities = []
        for chunk in self.chunks:
            doc = nlp(chunk["text"])
            # current_chunk_entities = []
            per_chunk_ent_freq = {}
            per_chunk_ent_metadata = {}
            for ent in doc.ents:
                entity_text = ent.text.strip()
                entity_text_norm = entity_text.lower()
                if len(entity_text) <= 1 or entity_text.isdigit():
                    continue
                
                if entity_text_norm in per_chunk_ent_freq:
                    per_chunk_ent_freq[entity_text_norm] += 1
                else:
                    per_chunk_ent_freq[entity_text_norm] = 1

                    per_chunk_ent_metadata[entity_text_norm] = {
                        "text_norm": entity_text_norm,
                        "text_raw": entity_text,
                        "label": ent.label_
                    }
            
            current_chunk_entities = []
            for et, count in per_chunk_ent_freq.items():
                metadata = per_chunk_ent_metadata[et]
                current_chunk_entities.append({
                    "text_norm": metadata["text_norm"],
                    "text_raw": metadata["text_raw"],
                    "label": metadata["label"],
                    "count": count
                })
            entities.append({
                "chunk_id": chunk["chunk_id"],
                "entities": current_chunk_entities
            })
        return entities
    
    def create_chunk_entities(self):
        entities = self.extract_entities()
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(CHUNK_ENTITIES_PATH, "w") as f:
            json.dump({"chunk_entities": entities}, f, indent=2)
        