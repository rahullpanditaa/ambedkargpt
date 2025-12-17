import json
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import networkx as nx
import pickle


DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
COMMUNITY_SUMMARIES_PATH = PROCESSED_DATA_DIR_PATH / "community_summaries.json"
COMMUNITY_EMBEDDINGS_PATH = PROCESSED_DATA_DIR_PATH / "community_embeddings.npy"
CHUNK_ENTITIES_PATH = PROCESSED_DATA_DIR_PATH / "chunk_entities.json"
KNOWLEDGE_GRAPH_PATH = PROCESSED_DATA_DIR_PATH / "knowledge_graph.pkl"


# hyperparameters
MAX_SEED_ENTITIES = 10        # only expand from top-N entities
MAX_NEIGHBORS_PER_ENTITY = 5 # prevent explosion
NEIGHBOR_DECAY = 0.6          # neighbors are weaker than seeds
MIN_NEIGHBOR_WEIGHT = 1       # ignore very weak edges

class LocalGraphRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def _get_entity_texts(self) -> list[str]:
        # need embedding for every entity
        with open(CHUNK_ENTITIES_PATH, "r") as f:
            chunk_entities: list[dict] = json.load(f)["chunk_entities"]
        entity_texts: list[str] = []
        for chunk in chunk_entities:
            entities = chunk["entities"]
            for ent in entities:
                entity_texts.append(ent["text_norm"])
        return entity_texts
    
    def _retrieve_entitities(self, user_query: str):
        query_embedding = self.model.encode([user_query])[0]

        entity_texts = self._get_entity_texts()
        entity_texts_embeddings = self.model.encode(entity_texts)

        query_entities_sim_scores = []

        for i, ent_emb in enumerate(entity_texts_embeddings):
            sim_score = _cosine_similarity(vec1=ent_emb, vec2=query_embedding)
            if sim_score >= 0.4:
                query_entities_sim_scores.append({
                    "entity": entity_texts[i],
                    "score": sim_score
                })

        # sort by similarity score desc
        sorted_scores = sorted(query_entities_sim_scores, key=lambda d: d["score"], reverse=True)
        # return sorted_scores

        # expand via graph neighbours
        with open(KNOWLEDGE_GRAPH_PATH, "rb") as f:
            knowledge_graph: nx.Graph = pickle.load(f)
    
        # for entity_score in sorted_scores:
        #     entity_neighbours = list(knowledge_graph.neighbors(entity_score))
            
        expanded_entity_scores = {}
        for item in sorted_scores[:MAX_SEED_ENTITIES]:
            entity = item["entity"]
            score = item["score"]

            # max score if entity reappears
            expanded_entity_scores[entity] = max(expanded_entity_scores.get(entity, 0.0), score)

        # expand - graph neigbours
        for item in sorted_scores[:MAX_SEED_ENTITIES]:
            seed_ent = item["entity"]
            seed_score = item["score"]

            if seed_ent not in knowledge_graph:
                continue

            neighbours = knowledge_graph[seed_ent]

            # sort neighbours by edge weight
            sorted_neighbours = sorted(neighbours.items(), key=lambda x: x[1].get("weight", 1), reverse=True)

            for neighbour, edge_data in sorted_neighbours[:MAX_NEIGHBORS_PER_ENTITY]:
                edge_weight = edge_data.get("weight", 1)

                if edge_weight < MIN_NEIGHBOR_WEIGHT:
                    continue

                neighbour_score = seed_score * NEIGHBOR_DECAY

                expanded_entity_scores[neighbour] = max(expanded_entity_scores.get(neighbour, 0.0), neighbour_score)

            expanded_query_entities = [
                {"entity": ent, "score": score}
                for ent, score in expanded_entity_scores.items()
            ]

            expanded_query_entities.sort(key=lambda d: d["score"], reverse=True)
            return expanded_query_entities
    

    
    def _expand_recall_via_graph_neighbours(self):
            ... 

    def _chunk_entity_similarity(self):
        # retrieve entities relevant to the query
        retrieved_entities = self._retrieve_entitities()

        active_entities = []
        entity_score_map = {}

        for entity_score in retrieved_entities:
            if entity_score["score"] > 0.6:
                active_entities.append(entity_score["entity"])
                entity_score_map[entity_score["entity"]] = entity_score["score"]

        if not active_entities:
            return []

        # entity -> chunk_ids map
        with open(CHUNK_ENTITIES_PATH, "r") as f:
            chunk_entities = json.load(f)["chunk_entities"]

        entity_to_chunk_ids_map = {}
        chunk_entity_freq_map = {}

        for chunk in chunk_entities:
            chunk_id = chunk["chunk_id"]
            chunk_entity_freq_map[chunk_id] = {}

            for entity in chunk["entities"]:
                entity_text = entity["text_norm"]
                count = entity.get("count", 1)

                # entity -> chunks
                if entity_text not in entity_to_chunk_ids_map:
                    entity_to_chunk_ids_map[entity_text] = []
                entity_to_chunk_ids_map[entity_text].append(chunk_id)

                # chunk -> entity freq
                chunk_entity_freq_map[chunk_id][entity_text] = count

        # collect candidate chunk ids
        candidate_chunk_ids = set()

        for active_entity in active_entities:
            if active_entity in entity_to_chunk_ids_map:
                for cid in entity_to_chunk_ids_map[active_entity]:
                    candidate_chunk_ids.add(cid)

        if not candidate_chunk_ids:
            return []

        # load chunk texts
        with open(PROCESSED_DATA_DIR_PATH / "chunks.json", "r") as f:
            chunks = json.load(f)["chunks"]

        chunk_id_to_text = {
            ch["chunk_id"]: ch["text"] for ch in chunks
        }

        # 
        scored_chunks = []

        for chunk_id in candidate_chunk_ids:
            score = 0.0

            #for each active entity in this chunk
            for entity in active_entities:
                if entity in chunk_entity_freq_map.get(chunk_id, {}):
                    entity_weight = entity_score_map[entity]
                    entity_freq = chunk_entity_freq_map[chunk_id][entity]

                    # contribution = entity relevance * frequency in chunk
                    score += entity_weight * entity_freq

            if score > 0:
                scored_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_id_to_text.get(chunk_id, ""),
                    "score": score
                })

        # sort by relevance
        scored_chunks.sort(key=lambda d: d["score"], reverse=True)

        return scored_chunks


    # def _chunk_entity_similarity(self):
    #     retrieved_entities = self._retrieve_entitities()

    #     # only keep entities with score above certain threshold
    #     active_entities = []
    #     for entity_score in retrieved_entities:
    #         if entity_score["score"] > 0.6:
    #             active_entities.append(entity_score["entity"])

    #     # need entity -> chunk ids mapping
    #     with open(CHUNK_ENTITIES_PATH, "r") as f:
    #         # list[dict], where dict keys -> chunk_id, entities: list[dict - text_norm]
    #         chunk_entities: list[dict] = json.load(f)["chunk_entities"]
    #     entity_to_chunk_ids_map = {}
    #     for chunk in chunk_entities:
    #         # current_chunk_entities = []
    #         for entity in chunk["entities"]:
    #             entity_text = entity["text_norm"]
    #             if entity_text not in entity_to_chunk_ids_map:
    #                 entity_to_chunk_ids_map[entity_text] = []
    #             entity_to_chunk_ids_map[entity_text].append(chunk["chunk_id"])
            
    #     # also need the chunk text itself
    #     with open(PROCESSED_DATA_DIR_PATH / "chunks.json", "r") as f:
    #         # list[dict] where dict keys - chunk_id, text , ...
    #         chunks = json.load(f)["chunks"]

    #     canditate_chunk_ids = set()
    #     for active_entity in active_entities:
    #         if active_entity in entity_to_chunk_ids_map:
    #             canditate_chunk_ids.add(*entity_to_chunk_ids_map[active_entity])

    #     candidate_chunks = []
    #     for chunk in chunks:
    #         if chunk["chunk_id"] in canditate_chunk_ids:
    #             candidate_chunks.append({
    #                 "chunk_id": chunk["chunk_id"],
    #                 "chunk_text": chunk["text"]
    #             })

    #     # compute relevance score for each candidate chunk


        
        

def _cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)