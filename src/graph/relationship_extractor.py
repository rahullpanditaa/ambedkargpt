import json
from pathlib import Path
from itertools import combinations

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
CHUNK_ENTITIES_PATH = PROCESSED_DATA_DIR_PATH / "chunk_entities.json"
ENTITY_RELATIONS_PATH = PROCESSED_DATA_DIR_PATH / "entity_relations.json"

class RelationshipExtractor:
    def __init__(self):
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(CHUNK_ENTITIES_PATH, "r") as f:
            entitites = json.load(f)["chunk_entities"]
        
        self.chunk_entities = list(filter(lambda d: len(d["entities"]) >= 2, entitites))
        
    def extract_relationships(self):
        edges = []

        for ch in self.chunk_entities:
            
            entitites = ch["entities"]

            en_texts = [e["text_norm"] for e in entitites]

            #           subsequences of length 2 generated from en_texts
            for a, b in combinations(en_texts, 2):
                src, trgt = sorted([a, b])

                edges.append({
                    "source": src,
                    "target": trgt,
                    "relation": "co_occurence",
                    "chunk_id": ch["chunk_id"],
                    "weight": 1
                })
        return edges
    
    def save_relationships(self):
        edges = self.extract_relationships()

        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(ENTITY_RELATIONS_PATH, "w") as f:
            json.dump({"edges": edges}, f, indent=2)