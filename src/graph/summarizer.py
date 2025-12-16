
import json
from pathlib import Path

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
# KNOWLEDGE_GRAPH_PATH = PROCESSED_DATA_DIR_PATH / "knowledge_graph.pkl"
ENTITY_COMMUNITY_PATH = PROCESSED_DATA_DIR_PATH / "entity_communities.json"

class CommunitySummarizer:
    def __init__(self):
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_DIR_PATH, "r") as f:
            entity_communities = json.load(f)
        self.entity_to_comm_id_map: dict = entity_communities["entity_communities"]
        self.comm_id_to_entities_map: dict = {}

        # for entity, comm_id in ent->id map:
        # if comm_id not in id->entities map 
        #     id->entities map[comm_id] = []
        # id -> entities map[comm_id] . append(entity)