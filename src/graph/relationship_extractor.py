"""
Relationship extraction for knowledge graph construction.

This module derives relationships between entities based on
co-occurrence within the same semantic chunk. Each pair of
entities appearing together in a chunk is treated as a
co-occurrence edge in the knowledge graph.

These relationships form the edge set used in downstream
graph construction and community detection.
"""

import json
from itertools import combinations
from src.utils.constants import (
    PROCESSED_DATA_DIR_PATH,
    CHUNK_ENTITIES_PATH,
    ENTITY_RELATIONS_PATH
)

class RelationshipExtractor:
    """
    Extracts relationships between entities based on chunk-level co-occurrence.

    For each semantic chunk containing two or more entities:
    - All unordered pairs of entities are generated
    - Each pair is treated as a co-occurrence relationship
    - Relationships are annotated with chunk provenance

    This produces a simple but effective relational structure
    suitable for building a knowledge graph.
    """
    def __init__(self):
        """
        Initialize the relationship extractor.

        Loads per-chunk entity data from disk and filters out
        chunks with fewer than two entities (since no relationships
        can be formed).
        """
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(CHUNK_ENTITIES_PATH, "r") as f:
            entitites = json.load(f)["chunk_entities"]
        
        self.chunk_entities = list(filter(lambda d: len(d["entities"]) >= 2, entitites))
        
    def _extract_relationships(self):
        """
        Extract co-occurrence relationships from chunk-level entities.

        For each chunk:
        - Generate all unordered pairs of entities
        - Normalize direction by sorting entity names
        - Assign a co-occurrence relationship type

        Returns:
            list[dict]: List of relationship edges with schema:
                {
                    "source": str,
                    "target": str,
                    "relation": str,
                    "chunk_id": int,
                    "weight": int
                }
        """
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
        """
        Extract relationships and persist them to disk.

        Side Effects:
            - Writes relationship edges to ENTITY_RELATIONS_PATH
        """
        edges = self._extract_relationships()

        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(ENTITY_RELATIONS_PATH, "w") as f:
            json.dump({"edges": edges}, f, indent=2)


def entity_relations_command():
    """
    CLI-style helper for extracting and saving entity relationships.
    """
    re = RelationshipExtractor()
    re.save_relationships()
    print("Entity relations saved to disk")