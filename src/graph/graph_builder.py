"""
Knowledge graph construction for the SemRAG pipeline.

This module builds a weighted, undirected knowledge graph from
entity co-occurrence relationships extracted at the chunk level.
The resulting graph is used for community detection and both
local and global Graph RAG retrieval.
"""

import json
import networkx as nx
import pickle
from src.utils.constants import (
    PROCESSED_DATA_DIR_PATH, 
    ENTITY_RELATIONS_PATH, 
    KNOWLEDGE_GRAPH_PATH
)

class GraphBuilder:
    """
    Builds a weighted knowledge graph from entity relationships.

    The graph is constructed as:
    - Nodes: normalized entity strings
    - Edges: co-occurrence relationships between entities
    - Edge weights: frequency of co-occurrence across chunks

    The resulting graph is undirected and weighted, suitable for
    community detection algorithms such as Louvain or Leiden.
    """
    def __init__(self):
        """
        Initialize the graph builder.

        Loads entity relationship edges from disk and prepares
        an empty NetworkX graph for construction.
        """
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(ENTITY_RELATIONS_PATH, "r") as f:
            relations = json.load(f)["edges"]

        self.relations = [{"source": rel["source"],
                           "target": rel["target"],
                           "weight": rel["weight"]}
                           for rel in relations]

    def _build_graph(self):
        graph = nx.Graph()
        """
        Construct the weighted knowledge graph.

        For each relationship:
        - If an edge already exists between two entities,
          increment its weight
        - Otherwise, create a new edge with the given weight

        Returns:
            networkx.Graph: The constructed knowledge graph
        """
        for rel in self.relations:
            src = rel["source"]
            trgt = rel["target"]
            # does an edge exist bw src and trgt ?
            # if no - create it, weight = 1
            # if yes - weight ++
            if graph.has_edge(src, trgt):
               graph.edges[(src, trgt)]["weight"] += rel["weight"]
            else:
                graph.add_edge(src, trgt, weight=rel["weight"])
        print("Graph built")
        print(f"- Number of nodes: {graph.number_of_nodes()}")
        print(f"- Number of edges: {graph.number_of_edges()}")
        return graph

    def save_graph(self):
        """
        Persist the constructed knowledge graph to disk.

        If the graph has not yet been built, it will be constructed
        before being saved.

        Side Effects:
            - Writes the graph object to KNOWLEDGE_GRAPH_PATH using pickle
        """
        graph = self._build_graph()

        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_GRAPH_PATH, "wb") as f:
            pickle.dump(graph, f)

def build_graph_command():
    gb = GraphBuilder()
    gb.save_graph()