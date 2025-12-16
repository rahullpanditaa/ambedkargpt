
import json
from pathlib import Path

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
# KNOWLEDGE_GRAPH_PATH = PROCESSED_DATA_DIR_PATH / "knowledge_graph.pkl"
ENTITY_COMMUNITY_PATH = PROCESSED_DATA_DIR_PATH / "entity_communities.json"
CHUNKS_OUTPUT_PATH = PROCESSED_DATA_DIR_PATH / "chunks.json"
CHUNK_ENTITIES_PATH = PROCESSED_DATA_DIR_PATH / "chunk_entities.json"

class CommunitySummarizer:
    def __init__(self):
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_DIR_PATH, "r") as f:
            entity_communities = json.load(f)
        self.entity_to_comm_id_map: dict = entity_communities["entity_communities"]
        self.comm_id_to_entities_map: dict = {}

        for entity, comm_id in self.entity_to_comm_id_map.items():
            if comm_id not in self.comm_id_to_entities_map.keys():
                self.comm_id_to_entities_map[comm_id] = []
            self.comm_id_to_entities_map[comm_id].append(entity)

    def collect_chunks_per_community(self):
        ch_ents = load_json(CHUNK_ENTITIES_PATH)
        # list of dict where dict - chunk_id, entities: list[dict - text_norm, text_raw, label, count]
        chunk_entities = ch_ents["chunk_entities"]

        # community id -> set of chunk ids
        community_id_chunks = []
        for ch in chunk_entities:
            com_id_to_chunk_id = {}
            for entity in ch["entities"]:
                if entity["text_norm"] in self.entity_to_comm_id_map:
                    chunk_com_id = self.entity_to_comm_id_map[entity["text_norm"]]
                    if chunk_com_id not in com_id_to_chunk_id:
                        com_id_to_chunk_id[chunk_com_id] = set()
                    com_id_to_chunk_id[chunk_com_id].add(f"chunk_{entity['chunk_id']}")
            community_id_chunks.append(com_id_to_chunk_id)

        return community_id_chunks


def load_json(file_path: Path):
    with open(file_path, "r") as f:
        result = json.load(f)
    return result