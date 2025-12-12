import json
import numpy as np
from src.utils.constants import (
    SEGMENTS_DISTANCES_PATH,
    BUFFER_MERGE_RESULTS_PATH,
    BOOK_SENTENCES_PATH,
    CHUNKS_OUTPUT_PATH
)
THETA = 0.30
MAX_TOKENS = 1024
SUBCHUNK_SIZE = 128
SUBCHUNK_OVERLAP = 32

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
    
    def _split_large_chunk(self, chunk: dict) -> list[dict]:

        if chunk["num_tokens"] <= MAX_TOKENS:
            return [chunk]

        text = chunk["text"]
        tokens = self.tokenizer.encode(text)

        subchunks = []
        start = 0

        while start < len(tokens):
            end = start + SUBCHUNK_SIZE
            window_tokens = tokens[start:end]
            sub_text = self.tokenizer.decode(window_tokens)

            subchunks.append({
                "sentence_indices": chunk["sentence_indices"],  # optional
                "text": sub_text,
                "num_tokens": len(window_tokens)
            })

            # overlap for continuity
            start = end - SUBCHUNK_OVERLAP

            if start < 0:
                break

        return subchunks
    
    def create_chunks(self) -> list[dict]:
        distances = self.segment_distances  

        chunks = []
        current_chunk_units = [0] 

        for i in range(len(distances)):
            d = distances[i]

            if d < THETA:
                current_chunk_units.append(i + 1)
            else:
                # semantic break --close chunk, start new one
                chunks.append(current_chunk_units)
                current_chunk_units = [i + 1]

        chunks.append(current_chunk_units)

        final_chunks = []
        chunk_id = 1

        for unit_indices in chunks:
            chunk_obj = self._reconstruct_chunk_from_sentences(unit_indices)

            # if chunk is large -- split
            subchunks = self._split_large_chunk(chunk_obj)

            for sc in subchunks:
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "text": sc["text"],
                    "sentence_indices": sc["sentence_indices"],
                    "num_tokens": sc["num_tokens"],
                    "source_units": unit_indices  # helpful for debugging
                })
                chunk_id += 1
                
        with open(CHUNKS_OUTPUT_PATH, "w") as f:
            json.dump({"chunks": final_chunks}, f, indent=2)

        return final_chunks