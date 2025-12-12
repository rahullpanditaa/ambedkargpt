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

    def _chunk_reconstruction_from_sentences(self, chunk_unit_indices: list[int]) -> dict:
        # convert merged-units into final chunk text
        all_sentences_indices = []

        for unit_index in chunk_unit_indices:
            unit = self.merged_units[unit_index]
            start = unit["start"]
            end = unit["end"]
            all_sentences_indices.extend(range(start, end+1))

        seen = set()
        ordered_sentences_idxs = []
        for idx in all_sentences_indices:
            if idx not in seen:
                seen.add(idx)
                ordered_sentences_idxs.append(idx)

        final_text = " ".join(self.sentences[i]["text"] for i in ordered_sentences_idxs)

        num_of_tokens = len(self.tokenizer.encode(final_text))

        return {
            "sentence_indices": ordered_sentences_idxs,
            "text": final_text,
            "num_tokens": num_of_tokens
        }