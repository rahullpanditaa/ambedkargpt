"""
Entity extraction stage for knowledge graph construction.

This module extracts named entities from semantic chunks produced
by the semantic chunking stage. Each chunk is processed independently
using spaCy NER, and entities are aggregated with per-chunk frequencies.

The output of this module is used as input for:
- Knowledge graph construction
- Local and global Graph RAG retrieval
"""
import json
import spacy
from pathlib import Path
from src.utils.constants import (
    PROCESSED_DATA_DIR_PATH,
    CHUNKS_OUTPUT_PATH,
    CHUNK_ENTITIES_PATH
)

# DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
# PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
# CHUNKS_OUTPUT_PATH = PROCESSED_DATA_DIR_PATH / "chunks.json"
# CHUNK_ENTITIES_PATH = PROCESSED_DATA_DIR_PATH / "chunk_entities.json"

nlp = spacy.load("en_core_web_sm")

class EntityExtractor:
    """
    Extracts named entities from semantic chunks.

    For each chunk:
    - Runs spaCy NER on the chunk text
    - Normalizes entity surface forms
    - Aggregates per-chunk entity frequencies
    - Preserves both raw and normalized representations

    The resulting entity lists form the basis for
    knowledge graph nodes and entity-centric retrieval.
    """
    def __init__(self):
        """
        Initialize the entity extractor.

        Loads semantic chunks from disk and ensures the
        processed data directory exists.
        """
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(CHUNKS_OUTPUT_PATH, "r") as f:
            chunks = json.load(f)
        # keys - chunk id, text, ... etc.
        self.chunks: list[dict] = chunks["chunks"]

    def _extract_entities(self) -> list[dict]:
        """
        Extract named entities from each semantic chunk.

        Entity extraction is performed independently per chunk.
        Entities are normalized (lowercased) for aggregation while
        preserving raw surface forms and NER labels.

        Returns:
            list[dict]: List of per-chunk entity mappings with schema:
                {
                    "chunk_id": int,
                    "entities": [
                        {
                            "text_norm": str,
                            "text_raw": str,
                            "label": str,
                            "count": int
                        },
                        ...
                    ]
                }
        """
        entities = []
        for chunk in self.chunks:
            doc = nlp(chunk["text"])
            
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
        """
        Extract entities and persist them to disk.

        Side Effects:
            - Writes entity extraction results to CHUNK_ENTITIES_PATH
        """
        entities = self._extract_entities()
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(CHUNK_ENTITIES_PATH, "w") as f:
            json.dump({"chunk_entities": entities}, f, indent=2)
        

def extract_entities_command():
    """
    CLI-style helper for running entity extraction manually.
    """
    ee = EntityExtractor()
    ee.create_chunk_entities()
    print("Extracted entities")