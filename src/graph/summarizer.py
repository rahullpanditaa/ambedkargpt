
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

    def _collect_chunks_per_community(self):
        ch_ents = load_json(CHUNK_ENTITIES_PATH)
        # list of dict where dict - chunk_id, entities: list[dict - text_norm, text_raw, label, count]
        chunk_entities = ch_ents["chunk_entities"]

        # community id -> set of chunk ids
        community_id_to_chunks = {}
        for chunk in chunk_entities:
            chunk_id = chunk["chunk_id"]
            current_chunk_in_communities = set()
            
            for entity in chunk["entities"]:
                if entity["text_norm"] in self.entity_to_comm_id_map:
                    chunk_community_id = self.entity_to_comm_id_map[entity["text_norm"]]
                    current_chunk_in_communities.add(chunk_community_id)
                    
            for community_id in current_chunk_in_communities:
                if community_id not in community_id_to_chunks:
                    community_id_to_chunks[community_id] = set()
                community_id_to_chunks[community_id].add(chunk_id)

        return community_id_to_chunks
    
    def select_representative_chunks(self):
        # for each community - select top chunks until reach arbitrarily chosen token budget
        # return community id -> list of chunk texts

        # load required data
        community_id_to_chunks = self._collect_chunks_per_community()

        chunks_data = load_json(CHUNKS_OUTPUT_PATH)["chunks"]
        chunk_entities_data = load_json(CHUNK_ENTITIES_PATH)["chunk_entities"]

        # lookup maps - chunk id -> chunk text
        chunk_id_to_text = {
            ch["chunk_id"]: ch["text"] for ch in chunks_data
        }

        # chunk id -> entities
        chunk_id_to_entities = {
            ch["chunk_id"]: ch["entities"] for ch in chunk_entities_data
        }

        tokens_per_community = 2500  # random
        selected_chunks_per_community = {}

        # process each community independently
        for community_id, chunk_ids in community_id_to_chunks.items():

            scored_chunks = []

            for chunk_id in chunk_ids:
                entities = chunk_id_to_entities.get(chunk_id, [])

                # sum of entity frequencies
                score = sum(ent["count"] for ent in entities)

                if score == 0:
                    continue

                scored_chunks.append({
                    "chunk_id": chunk_id,
                    "score": score,
                    "text": chunk_id_to_text[chunk_id]
                })

            # sort chunks by descending score
            scored_chunks.sort(key=lambda d: d["score"], reverse=True)

            selected_chunks = []
            used_tokens = 0

            for item in scored_chunks:
                chunk_text = item["text"]

                chunk_tokens = len(chunk_text.split())

                if used_tokens + chunk_tokens > tokens_per_community:
                    break

                selected_chunks.append({
                    "chunk_id": item["chunk_id"],
                    "text": chunk_text
                })

                used_tokens += chunk_tokens

            if not selected_chunks and scored_chunks:
                fallback = scored_chunks[0]
                selected_chunks.append({
                    "chunk_id": fallback["chunk_id"],
                    "text": fallback["text"]
                })

            selected_chunks_per_community[community_id] = selected_chunks

        return selected_chunks_per_community




def load_json(file_path: Path):
    with open(file_path, "r") as f:
        result = json.load(f)
    return result