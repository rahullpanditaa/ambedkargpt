"""
Semantic chunking implementation for the SemRAG pipeline.

This module implements Algorithm 1 from the SemRAG paper:
- Uses cosine distance between adjacent buffer-merged units
- Groups merged units into semantically coherent chunks
- Enforces token limits via overlapping sub-chunking

The output of this module represents the final semantic chunks (C),
which are used downstream for knowledge graph construction and retrieval.
"""

import json
import numpy as np
from transformers import AutoTokenizer
from src.utils.constants import (
    SEGMENTS_DISTANCES_PATH,
    BUFFER_MERGE_RESULTS_PATH,
    BOOK_SENTENCES_PATH,
    CHUNKS_OUTPUT_PATH,
    THETA,
    MAX_TOKENS,
    SUBCHUNK_SIZE,
    SUBCHUNK_OVERLAP
)

# # semantic chunking hyperparameters
# THETA = 0.30
# MAX_TOKENS = 1024
# SUBCHUNK_SIZE = 128
# SUBCHUNK_OVERLAP = 32



class SemanticChunking:
    """
    Constructs semantic chunks from buffer-merged sentence units.

    This class consumes:
    - Buffer-merged units (Ŝ)
    - Pairwise cosine distances between adjacent merged units

    And produces:
    - Semantically coherent chunks (C), optionally split into
      overlapping sub-chunks to satisfy token constraints.

    The implementation closely follows Algorithm 1 from the
    SemRAG research paper.
    """
    def __init__(self):
        """
        Initialize the semantic chunking pipeline.

        Loads all required intermediate artifacts from disk:
        - Cosine distances between adjacent merged units
        - Buffer-merged units
        - Original sentence metadata (for reconstruction)

        Also initializes a tokenizer used for token counting
        and sub-chunk splitting.
        """
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
        """
        Reconstruct a semantic chunk from merged-unit indices.

        Given a list of merged-unit indices that belong to the same
        semantic chunk, this method:
        - Collects all contributing sentence indices
        - Deduplicates them while preserving original order
        - Reconstructs the final chunk text
        - Computes token count

        Args:
            chunk_unit_indices (list[int]): Indices of merged units
                                            forming a semantic chunk

        Returns:
            dict: Chunk representation with schema:
                {
                    "sentence_indices": list[int],
                    "text": str,
                    "num_tokens": int
                }
        """
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
        """
        Split an oversized chunk into overlapping sub-chunks.

        If a chunk exceeds MAX_TOKENS, it is split into smaller
        overlapping sub-chunks of size SUBCHUNK_SIZE with
        SUBCHUNK_OVERLAP tokens of overlap to preserve continuity.

        Args:
            chunk (dict): Chunk object containing text and token count

        Returns:
            list[dict]: One or more chunk / sub-chunk objects
        """
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
        """
        Construct semantic chunks using cosine distance thresholding.

        Implements the core logic of Algorithm 1:
        - Adjacent merged units are grouped if their cosine distance
          is below THETA
        - A semantic break starts a new chunk
        - Oversized chunks are split into overlapping sub-chunks

        Returns:
            list[dict]: Final list of semantic chunks with schema:
                {
                    "chunk_id": int,
                    "text": str,
                    "sentence_indices": list[int],
                    "num_tokens": int,
                    "source_units": list[int]
                }

        Side Effects:
            - Writes chunk data to CHUNKS_OUTPUT_PATH
        """
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
            chunk_obj = self._chunk_reconstruction_from_sentences(unit_indices)

            # if chunk is large -- split
            subchunks = self._split_large_chunk(chunk_obj)

            for sc in subchunks:
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "text": sc["text"],
                    "sentence_indices": sc["sentence_indices"],
                    "num_tokens": sc["num_tokens"],
                    "source_units": unit_indices 
                })
                chunk_id += 1

        with open(CHUNKS_OUTPUT_PATH, "w") as f:
            json.dump({"chunks": final_chunks}, f, indent=2)

        return final_chunks
    
def create_chunks_command(limit: int=5):
    """
    CLI-style helper for running semantic chunking manually.

    Args:
        limit (int): Number of chunks to print for inspection
    """
    sc = SemanticChunking()
    print(f"Creating chunks from 'data/Ambedkar_book.pdf'...")
    chunks = sc.create_chunks()
    print(f"Printing first {limit} chunks...")
    for i, ch in enumerate(chunks[:limit], 1):
        print(f"{i}. ({ch['chunk_id']}) Text: {ch['text'][:100]}...")
