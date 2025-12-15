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

    def build_graph(self):
        for rel in self.relations:
            src = rel["source"]
            trgt = rel["target"]
            # does an edge exist bw src and trgt ?
            # if no - create it, weight = 1
            # if yes - weight ++
            if self.graph.has_edge(src, trgt):
                self.graph.edges[(src, trgt)]["weight"] += 1
            else:
                self.graph.add_edge(src, trgt, weight=1)

    def save_graph(self):
        if len(list(self.graph.edges)) == 0:
            self.build_graph()

        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_DIR_PATH, "wb") as f:
            pickle.dump(self.graph, f)