# import json
# import numpy as np
# from src.utils.constants import (
#     BUFFER_MERGE_RESULTS_PATH,
#     SEGMENTS_EMBEDDINGS_PATH,
#     SEGMENTS_DISTANCES_PATH
# )
# from sentence_transformers import SentenceTransformer

# class MergedUnitsEmbedder:
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         with open(BUFFER_MERGE_RESULTS_PATH, "r") as f:
#             buffer_merge_results = json.load(f)
#         self.merged_units = buffer_merge_results["buffer_merge_results"]
#         self.model = SentenceTransformer(model_name_or_path=model_name)
#         self.segment_embeddings = None

#     def _build_embeddings(self) -> np.ndarray:
#         all_units_str = []
#         for unit in self.merged_units:
#             all_units_str.append(unit["text"])
#         self.segment_embeddings = self.model.encode(all_units_str, show_progress_bar=True)
#         np.save(SEGMENTS_EMBEDDINGS_PATH, self.segment_embeddings)
#         return self.segment_embeddings
    
#     def load_or_create_embeddings(self) -> np.ndarray:
#         if SEGMENTS_EMBEDDINGS_PATH.exists():
#             self.segment_embeddings = np.load(SEGMENTS_EMBEDDINGS_PATH)
#             if len(self.segment_embeddings) == len(self.merged_units):
#                 return self.segment_embeddings
#         # Algorithm 1 step 4
#         return self._build_embeddings()
    
#     def compute_cosine_distance(self):
#         if self.segment_embeddings is None or len(self.segment_embeddings) == 0:
#             raise ValueError("No segment embeddings loaded. Use 'embed-segments' command")
#         if SEGMENTS_DISTANCES_PATH.exists():
#             print(f"Cosine distances already calculated. File at '{SEGMENTS_DISTANCES_PATH.name}'")
#             return
        
#         cos_distances = []
#         for i in range(len(self.segment_embeddings) - 1):
#             d = 1 - _cosine_similarity(self.segment_embeddings[i],
#                                        self.segment_embeddings[i+1])
#             # Algorithm 1 steps 5,6
#             cos_distances.append(d)

#         numpy_arr = np.array(cos_distances, dtype=np.float32)
#         np.save(SEGMENTS_DISTANCES_PATH, numpy_arr)

#     def distances_inspection(self):
#         distances = np.load(SEGMENTS_DISTANCES_PATH)
#         print("Inspecting calculated cosine distances:")
#         print(f"- Mean: {np.mean(distances):.4f}")
#         print(f"- Standard deviation: {np.std(distances):.4f}")
#         print(f"- Maximum distance: {np.amax(distances):.4f}")
#         print(f"- Minimum distance: {np.amin(distances):.4f}")
        

# def embed_segments_command():
#     em = MergedUnitsEmbedder()
#     print("Generating vector embeddings for merged units (results of BufferMerge)...")
#     embeddings = em.load_or_create_embeddings()
#     print("Embeddings generated!!")
#     print(f"Embeddings of shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

# def compute_cosine_distances_command():
#     em = MergedUnitsEmbedder()
#     print("Computing cosine distances between consecutive segment embeddings...")
#     em.load_or_create_embeddings()
#     em.compute_cosine_distance()
#     print(f"- Distances computed. Saved to 'data/processed/{SEGMENTS_DISTANCES_PATH.name}'")

# def inspect_distances_command():
#     em = MergedUnitsEmbedder()
#     em.load_or_create_embeddings()
#     em.distances_inspection()

# def _cosine_similarity(vec1, vec2):
#     dot_product = np.dot(vec1, vec2)
#     norm1 = np.linalg.norm(vec1)
#     norm2 = np.linalg.norm(vec2)

#     if norm1 == 0 or norm2 == 0:
#         return 0.0
    
#     return dot_product / (norm1 * norm2)