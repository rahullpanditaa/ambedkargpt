"""
Community detection for the knowledge graph (SemRAG).

This module applies the Leiden community detection algorithm
to the entity co-occurrence knowledge graph. Detected communities
represent thematically related groups of entities and are used
in the Global Graph RAG retrieval step (Equation 5 in the SemRAG paper).
"""

import json
import pickle
import networkx as nx
import igraph as ig
from src.utils.constants import (
    PROCESSED_DATA_DIR_PATH,
    KNOWLEDGE_GRAPH_PATH,
    ENTITY_COMMUNITY_PATH
)

class CommunityDetector:
    """
    Detects communities in the knowledge graph using the Leiden algorithm.

    This class:
    - Loads the NetworkX knowledge graph
    - Converts it into an iGraph-compatible format
    - Runs Leiden community detection using edge weights
    - Produces a mapping from entity → community ID

    The resulting communities form the basis for Global Graph RAG search.
    """
    def __init__(self):
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_GRAPH_PATH, "rb") as f:
            self.old_format_graph: nx.Graph = pickle.load(f)
        self.graph_for_leiden = None
        self.entity_to_index_map = {}
        self.index_to_entity_map = {}
        self.edge_weights = []

    def _map_entities_to_integers(self) -> dict:
        """
        Assign integer indices to graph nodes.

        iGraph requires vertices to be represented as integer indices.
        This method builds bidirectional mappings between:
        - entity string ↔ integer index
        """
        for id, node in enumerate(list(self.old_format_graph.nodes)):
            self.entity_to_index_map[node] = id
            self.index_to_entity_map[id] = node
    
    def _convert_graph_edges_format(self):
        """
        Assign integer indices to graph nodes.

        iGraph requires vertices to be represented as integer indices.
        This method builds bidirectional mappings between:
        - entity string ↔ integer index
        """
        if not self.entity_to_index_map or not self.index_to_entity_map:
            self._map_entities_to_integers()
            
        new_graph_data = []
        new_graph_edges = []
        new_graph_weights = []
        for edge in self.old_format_graph.edges.data("weight"):
            src_ent = edge[0]
            trgt_ent = edge[1]
            weight = edge[2]
            src_idx = self.entity_to_index_map[src_ent]
            trgt_idx = self.entity_to_index_map[trgt_ent]
            new_graph_data.append({
                "source_index": src_idx,
                "target_index": trgt_idx,
                "weight": weight
            })
            new_graph_edges.append((src_idx, trgt_idx))
            new_graph_weights.append(weight)

        self.edge_weights = new_graph_weights
        return {
            "edges": new_graph_edges,
            "weights": new_graph_weights,
            "number_of_vertices": len(self.old_format_graph.nodes)
        }
    
    def _build_graph(self) -> ig.Graph:
        """
        Build an iGraph representation of the knowledge graph.

        This converts the NetworkX graph into an iGraph.Graph
        with weighted edges, which is required for Leiden clustering.
        """
        new_graph_data = self._convert_graph_edges_format()
        n = new_graph_data["number_of_vertices"]
        edges = new_graph_data["edges"]
        weights = new_graph_data["weights"]
        leiden_graph = ig.Graph(n=n, edges=edges, edge_attrs={"weight": weights})
        return leiden_graph

    def run_leiden(self):
        """
        Run the Leiden community detection algorithm.

        Produces a mapping from entity string → community ID
        and persists the result to disk.

        Side Effects:
            - Writes entity-to-community mapping to ENTITY_COMMUNITY_PATH
        """
        graph_for_leiden = self._build_graph()

        leiden = graph_for_leiden.community_leiden(weights=self.edge_weights)
        mem_vector = leiden.membership

        entity_to_community_map = {}
        for i, community_id in enumerate(mem_vector):
            entity = self.index_to_entity_map[i]
            entity_to_community_map[entity] = community_id

        with open(ENTITY_COMMUNITY_PATH, "w") as f:
            json.dump({"entity_communities": entity_to_community_map}, f,)

        
def run_leiden_command():
    """
    CLI-style helper for running Leiden community detection.
    """
    cd = CommunityDetector()
    cd.run_leiden()
    print("Ran Leiden algorithm successfully.")