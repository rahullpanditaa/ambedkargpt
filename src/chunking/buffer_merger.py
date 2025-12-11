import json

from src.utils.constants import (
    BOOK_SENTENCES_PATH, B, BUFFER_MERGE_RESULTS_PATH
)
class BufferMerge:
    def __init__(self):
        with open(BOOK_SENTENCES_PATH, "r") as f:
            sentences = json.load(f)
        self.sentences = sentences["sentences"]

    def buffer_merge(self, buffer_size: int=B):
        merged_units = []
        unit_id = 1
        for sent_idx, sentence in enumerate(self.sentences):
            # first_sent_idx = sent_idx
            # last_sent_idx = sent_idx + buffer_size
            # if sent_idx - buffer_size >= 0:
            #     first_sent_idx = sent_idx - buffer_size

            # if last_sent_idx >= len(self.sentences):
            #     last_sent_idx = len(self.sentences) - 1

            first_sent_idx = max(0, sent_idx - buffer_size)
            last_sent_idx = min(len(self.sentences) - 1, sent_idx + buffer_size) 
            
            # text = ""
            # for i in range(first_sent_idx, last_sent_idx + 1):
            #     text += self.sentences[i]["text"] + " "
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
        return merged_units[:20]
            
def buffer_merge_command(b: int=B):
    print("Performing BufferMerge on 'data/sentences.json'...")
    bm = BufferMerge()     
    merged_units = bm.buffer_merge()
    print("BufferMerge sucessful!!")
    print(f"First {len(merged_units)} merged units created:")
    for i, unit in enumerate(merged_units):
        print(f"{i}. {unit['text'][:50]}...")
        print(f"Starting sentence: {unit['start']}, Ending sentence: {unit['end']}")
        print()


            