import json
from pathlib import Path
import networkx as nx
import pickle

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
# CHUNK_ENTITIES_PATH = PROCESSED_DATA_DIR_PATH / "chunk_entities.json"
ENTITY_RELATIONS_PATH = PROCESSED_DATA_DIR_PATH / "entity_relations.json"
KNOWLEDGE_GRAPH_PATH = PROCESSED_DATA_DIR_PATH / "knowledge_graph.pkl"

class GraphBuilder:
    def __init__(self):
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(ENTITY_RELATIONS_PATH, "r") as f:
            relations = json.load(f)["edges"]

        self.relations = [{"source": rel["source"],
                           "target": rel["target"],
                           "weight": rel["weight"]}
                           for rel in relations]
        self.graph = nx.Graph()
        self._graph_built = False

    def build_graph(self):
        for rel in self.relations:
            src = rel["source"]
            trgt = rel["target"]
            # does an edge exist bw src and trgt ?
            # if no - create it, weight = 1
            # if yes - weight ++
            if self.graph.has_edge(src, trgt):
                self.graph.edges[(src, trgt)]["weight"] += rel["weight"]
            else:
                self.graph.add_edge(src, trgt, weight=rel["weight"])
        self._graph_built = True
        print("Graph built")
        print(f"- Number of nodes: {self.graph.number_of_nodes()}")
        print(f"- Number of edges: {self.graph.number_of_edges()}")


    def save_graph(self):
        if not self._graph_built:
            self.build_graph()
            self._graph_built = True

        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)