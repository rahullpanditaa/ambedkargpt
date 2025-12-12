import json
import numpy as np
from src.utils.constants import (
    SEGMENTS_DISTANCES_PATH,
    BUFFER_MERGE_RESULTS_PATH,
    BOOK_SENTENCES_PATH
)
THETA = 0.30
from transformers import AutoTokenizer

class SemanticChunking:
    def __init__(self):
        # d[i] - distance bw merged unit i and i+1
        self.segment_distances = np.load(SEGMENTS_DISTANCES_PATH)
        
        with open(BUFFER_MERGE_RESULTS_PATH, "r") as f:
            merged_units = json.load(f)
        self.merged_units = merged_units["buffer_merge_results"]

        with open(BOOK_SENTENCES_PATH, "r") as f:
            sentences = json.load(f)
        self.sentences = sentences["sentences"]

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def create_chunks(self):
        current_chunk = ""
        current_chunk += self.merged_units[0]["text"]
        
        chunks = []
        for i, unit in enumerate(self.merged_units, 1):
            if self.segment_distances[i] < THETA:
                current_chunk += unit["text"]
            else:
                chunks.append({
                    "id": i,
                    "chunk": current_chunk
                })
                current_chunk = unit["text"]
        
        return chunks