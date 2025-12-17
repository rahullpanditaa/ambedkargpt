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
            
    

    
    def _expand_recall_via_graph_neighbours(self):
        ... 
        

def _cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)