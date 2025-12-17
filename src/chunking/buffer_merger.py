"""
Buffer merging stage for semantic chunking (SemRAG Algorithm 1).

This module implements the buffer merging step described in the SemRAG paper,
where each sentence is expanded with a fixed window of neighboring sentences
to preserve local contextual continuity before embedding.

Formally, this corresponds to the construction of Ŝ (S-hat) in Algorithm 1.
"""

import json
from src.utils.constants import (
    PROCESSED_DATA_DIR_PATH, 
    BOOK_SENTENCES_PATH, 
    B, 
    BUFFER_MERGE_RESULTS_PATH
)

class BufferMerge:
    """
    Performs buffer-based contextual merging over sentence-level units.

    Given a sequence of sentences, this class constructs overlapping
    merged units by expanding each sentence with `buffer_size` neighboring
    sentences on both sides.

    Each merged unit:
    - Retains sentence index boundaries
    - Tracks contributing sentence IDs
    - Produces normalized merged text

    The resulting merged units are used as input for
    embedding and cosine similarity computation in semantic chunking.
    """
    def __init__(self):
        """
        Initialize the buffer merger.

        Side Effects:
            - Loads sentence data from BOOK_SENTENCES_PATH
        """
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(BOOK_SENTENCES_PATH, "r") as f:
            sentences = json.load(f)
        self.sentences = sentences["sentences"]

    def buffer_merge(self, buffer_size: int=B):
        """
        Perform buffer merging over the sentence sequence.

        For each sentence index `i`, a merged unit is constructed by
        concatenating sentences in the range:
            [i - buffer_size, i + buffer_size]

        Boundary conditions are handled by clamping indices to valid
        sentence positions.

        Args:
            buffer_size (int): Number of neighboring sentences to include
                               on each side of the central sentence.

        Returns:
            list[dict]: List of merged units with schema:
                {
                    "id": int,
                    "start": int,
                    "end": int,
                    "sentence_ids": list[str],
                    "text": str,
                    "character_count": int
                }

        Side Effects:
            - Writes merged units to BUFFER_MERGE_RESULTS_PATH

        Notes:
            - This corresponds to Ŝ (S-hat) in Algorithm 1 of the SemRAG paper.
            - Merged units are overlapping by construction.
        """
        merged_units = []
        unit_id = 1
        for sent_idx, _ in enumerate(self.sentences):

            first_sent_idx = max(0, sent_idx - buffer_size)
            last_sent_idx = min(len(self.sentences) - 1, sent_idx + buffer_size) 
            
            texts = [self.sentences[i]["text"] for i in range(first_sent_idx, last_sent_idx + 1)]
            text = " ".join(texts).strip()
            text = " ".join(text.split())

            sentence_ids = [self.sentences[i]["id"] for i in range(first_sent_idx, last_sent_idx + 1)]

            merged_units.append({
                "id": unit_id,
                "start": first_sent_idx,
                "end": last_sent_idx,
                "sentence_ids": sentence_ids,
                "text": text,
                "character_count": len(text)
            })
            unit_id += 1
        
        with open(BUFFER_MERGE_RESULTS_PATH, "w") as f:
            json.dump({"buffer_merge_results": merged_units}, f, indent=2)
        # S hat in algorithm 1 in SemRAG paper
        return merged_units

            
# def buffer_merge_command(b: int=B):
#     """
#     CLI-style helper function for running buffer merge manually.

#     Intended for debugging and inspection rather than pipeline usage.

#     Args:
#         b (int): Buffer size for sentence expansion
#     """
     
#     print("Performing BufferMerge on 'data/sentences.json'...")
#     bm = BufferMerge()     
#     merged_units = bm.buffer_merge(buffer_size=b)
#     print("BufferMerge sucessful!!")
#     print(f"First {len(merged_units)} merged units created:")
#     for i, unit in enumerate(merged_units):
#         print(f"{i}. {unit['text'][:50]}...")
#         print(f"Starting sentence: {unit['start']}, Ending sentence: {unit['end']}")
#         print()


            