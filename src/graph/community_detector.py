import json
import pickle
from pathlib import Path
import networkx as nx

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
KNOWLEDGE_GRAPH_PATH = PROCESSED_DATA_DIR_PATH / "knowledge_graph.pkl"


class CommunityDetector:
    def __init__(self):
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_GRAPH_PATH, "rb") as f:
            self.graph: nx.Graph = pickle.load(f)
        

    def _map_entities_to_integers(self) -> dict:
        ent_idx_map = {}
        for id, node in enumerate(list(self.graph.nodes)):
            # ent_idx_map.append({
            #     "entity": node,
            #     "index": id
            # })
            ent_idx_map.append({
                node: id,
                id: node
            })
            ent_idx_map[node] = id
            ent_idx_map[id] = node

        return ent_idx_map
    
    def _convert_graph_edges_format(self):
        entities_to_index_map = self._map_entities_to_integers()
        new_graph_data = []
        for edge in self.graph.edges.data("weight"):
            src_ent = edge[0]
            trgt_ent = edge[1]
            weight = edge[2]
            src_idx = entities_to_index_map[src_ent]
            trgt_idx = entities_to_index_map[trgt_ent]
            new_graph_data.append({
                "source_index": src_idx,
                "target_index": trgt_idx,
                "weight": weight
            })

        return new_graph_data


    

        