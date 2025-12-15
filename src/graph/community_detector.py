import json
import pickle
from pathlib import Path
import networkx as nx
import igraph as ig

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
KNOWLEDGE_GRAPH_PATH = PROCESSED_DATA_DIR_PATH / "knowledge_graph.pkl"
ENTITY_COMMUNITY_PATH = PROCESSED_DATA_DIR_PATH / "entity_communities.json"

class CommunityDetector:
    def __init__(self):
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_GRAPH_PATH, "rb") as f:
            self.old_format_graph: nx.Graph = pickle.load(f)
        self.graph_for_leiden = None
        self.entity_to_index_map = {}
        self.index_to_entity_map = {}
        self.edge_weights = []

    def _map_entities_to_integers(self) -> dict:
        for id, node in enumerate(list(self.old_format_graph.nodes)):
            # ent_idx_map.append({
            #     "entity": node,
            #     "index": id
            # })
            # ent_idx_map.append({
            #     node: id,
            #     id: node
            # })
            # ent_idx_map[node] = id
            # ent_idx_map[id] = node
            self.entity_to_index_map[node] = id
            self.index_to_entity_map[id] = node

        # return ent_idx_map
    
    def _convert_graph_edges_format(self):
        if not self.entity_to_index_map or not self.index_to_entity_map:
            self._map_entities_to_integers()
            
        new_graph_data = []
        new_graph_edges = []
        new_graph_weights = []
        # seen_entities = set()
        for edge in self.old_format_graph.edges.data("weight"):
            src_ent = edge[0]
            trgt_ent = edge[1]
            # seen_entities.add(src_ent)
            # seen_entities.add(trgt_ent)
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
    
    def build_graph(self):
        new_graph_data = self._convert_graph_edges_format()
        n = new_graph_data["number_of_vertices"]
        edges = new_graph_data["edges"]
        weights = new_graph_data["weights"]
        self.graph_for_leiden = ig.Graph(n=n, edges=edges, edge_attrs={"weight": weights})

    def run_leiden(self):
        if not self.graph_for_leiden:
            self.build_graph()

        leiden = self.graph_for_leiden.community_leiden(weights=self.edge_weights)
        mem_vector = leiden.membership

        entity_to_community_map = {}
        for i, community_id in enumerate(mem_vector):
            entity = self.index_to_entity_map[i]
            entity_to_community_map[entity] = community_id

        with open(ENTITY_COMMUNITY_PATH, "w") as f:
            json.dump({"entity_communities": entity_to_community_map}, f,)

        
def run_leiden_command():
    cd = CommunityDetector()
    cd.run_leiden()
    print("Ran Leiden algorithm successfully.")