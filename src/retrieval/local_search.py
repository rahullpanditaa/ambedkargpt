"""
Local Graph RAG (SemRAG) retrieval implementation.

This module implements the Local Graph RAG retrieval strategy
described in Equation (4) of the SemRAG paper. The approach
retrieves relevant entities for a user query, expands them via
graph neighborhoods, and scores semantic chunks based on
entity relevance and frequency.

The output is a ranked list of chunks most relevant to the query,
grounded in the local structure of the knowledge graph.
"""
import json
import numpy as np
import networkx as nx
import pickle
from sentence_transformers import SentenceTransformer
from src.utils.constants import (
    PROCESSED_DATA_DIR_PATH,
    CHUNK_ENTITIES_PATH, 
    KNOWLEDGE_GRAPH_PATH,
    LOCAL_SEARCH_RESULTS_PATH,
    MAX_SEED_ENTITIES, 
    MAX_NEIGHBORS_PER_ENTITY, 
    NEIGHBOR_DECAY,
    MIN_NEIGHBOR_WEIGHT,
    MIN_ENTITY_RELEVANCE_SCORE,
    MIN_SIMILARIY_SCORE_QUERY_ENTITY
)

class LocalGraphRAG:
    """
    Local Graph RAG retriever.

    This class performs query-time retrieval using:
    - Semantic similarity between query and entities
    - Graph-based expansion over the entity co-occurrence graph
    - Chunk scoring based on entity relevance and frequency

    The implementation corresponds directly to Equation (4)
    in the SemRAG research paper.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name_or_path=model_name)
        self.entity_texts: list[str] = []
        self.entity_texts_embeddings: np.ndarray = None

    def _get_entity_texts(self) -> list[str]:
        """
        Load and deduplicate all entities from chunk-level entity data.

        Reads entity surface forms from CHUNK_ENTITIES_PATH, removes
        duplicates, and caches the resulting list. Any existing
        entity embedding cache is invalidated.

        Returns:
            list[str]: List of unique normalized entity strings
        """
        # need embedding for every entity
        with open(CHUNK_ENTITIES_PATH, "r") as f:
            chunk_entities: list[dict] = json.load(f)["chunk_entities"]
        
        entity_texts = set()
        for chunk in chunk_entities:
            for entity in chunk["entities"]:
                entity_texts.add(entity["text_norm"])

        self.entity_texts = list(entity_texts)
        self.entity_texts_embeddings = None
        return self.entity_texts

    
    def _retrieve_entitities(self, user_query: str) -> list[dict]:
        """
        Retrieve entities semantically similar to the user query.

        Computes cosine similarity between the query embedding and
        all cached entity embeddings, filtering by a minimum
        similarity threshold.

        Args:
            user_query (str): User query string

        Returns:
            list[dict]: Sorted list of entities with similarity scores:
                {
                    "entity": str,
                    "score": float
                }
        """
        query_embedding = self.model.encode([user_query])[0]

        if not self.entity_texts:
            self._get_entity_texts()

        if self.entity_texts_embeddings is None:
            self.entity_texts_embeddings = self.model.encode(self.entity_texts)
        

        query_entities_sim_scores = []

        for i, ent_emb in enumerate(self.entity_texts_embeddings):
            sim_score = _cosine_similarity(vec1=ent_emb, vec2=query_embedding)
            if sim_score >= MIN_SIMILARIY_SCORE_QUERY_ENTITY:
                query_entities_sim_scores.append({
                    "entity": self.entity_texts[i],
                    "score": sim_score
                })

        # sort by similarity score desc
        sorted_scores = sorted(query_entities_sim_scores, key=lambda d: d["score"], reverse=True)
        return sorted_scores
    
    def _expand_recall_via_graph_neighbours(self, user_query: str):
        """
        Expand query-relevant entities using graph neighborhood traversal.

        Starts from top query-matched entities and propagates relevance
        scores to neighboring entities in the knowledge graph using
        edge weights and decay.

        Args:
            user_query (str): User query string

        Returns:
            list[dict]: Expanded list of entities with propagated scores
        """
        sorted_scores = self._retrieve_entitities(user_query=user_query)

        try:
            with open(KNOWLEDGE_GRAPH_PATH, "rb") as f:
                knowledge_graph: nx.Graph = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Knowledge graph missing or unreadable: {KNOWLEDGE_GRAPH_PATH.name}") from e
        
        expanded_entity_scores: dict[str, float] = {}
        for item in sorted_scores[:MAX_SEED_ENTITIES]:
            expanded_entity_scores[item["entity"]] = item["score"]

            # max score if entity reappears
            # expanded_entity_scores[entity] = max(expanded_entity_scores.get(entity, 0.0), score)

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
    
    def chunk_entity_similarity(self, user_query: str, k:int = 20):
        """
        Score semantic chunks using Local Graph RAG (Equation 4).

        Each chunk is scored by summing the product of:
        - entity relevance score (after graph expansion)
        - entity frequency within the chunk

        Args:
            user_query (str): User query string
            k (int): Number of top chunks to return

        Returns:
            list[dict]: Top-k ranked chunks with relevance scores
        """
        # retrieve entities relevant to the query
        retrieved_entities = self._expand_recall_via_graph_neighbours(user_query=user_query)

        active_entities = []
        entity_score_map = {}

        for entity in retrieved_entities:
            if entity["score"] > MIN_ENTITY_RELEVANCE_SCORE:
                active_entities.append(entity["entity"])
                entity_score_map[entity["entity"]] = entity["score"]

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

                entity_to_chunk_ids_map.setdefault(entity_text, set()).add(chunk_id)
                chunk_entity_freq_map[chunk_id][entity_text] = count

        # collect candidate chunk ids
        candidate_chunk_ids = set()

        for active_entity in active_entities:
            candidate_chunk_ids |= entity_to_chunk_ids_map.get(active_entity, set())

        if not candidate_chunk_ids:
            return []

        # load chunk texts
        with open(PROCESSED_DATA_DIR_PATH / "chunks.json", "r") as f:
            chunks = json.load(f)["chunks"]

        chunk_id_to_text = {
            ch["chunk_id"]: ch["text"] for ch in chunks
        }

        scored_chunks = []

        for cid in candidate_chunk_ids:
            score = 0.0
            for ent in active_entities:
                if ent in chunk_entity_freq_map[cid]:
                    score += (
                        entity_score_map[ent] *
                        chunk_entity_freq_map[cid][ent]
                    )

            if score > 0:
                scored_chunks.append({
                    "chunk_id": cid,
                    "chunk_text": chunk_id_to_text.get(cid, ""),
                    "score": score
                })

        scored_chunks.sort(key=lambda d: d["score"], reverse=True)

        with open(LOCAL_SEARCH_RESULTS_PATH, "w") as f:
            json.dump({"local_search_results": scored_chunks}, f, indent=2)

        return scored_chunks[:k]     

def _cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)