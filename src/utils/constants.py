from pathlib import Path
from pdfminer.high_level import extract_text

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
AMBEDKAR_BOOK_PATH =  DATA_DIR_PATH / "Ambedkar_book.pdf"
RAW_BOOK_TEXT = extract_text(AMBEDKAR_BOOK_PATH)

PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
BOOK_PARAGRAPHS_PATH = PROCESSED_DATA_DIR_PATH / "paragraphs.json"
BOOK_SENTENCES_PATH = PROCESSED_DATA_DIR_PATH / "sentences.json"

# Buffer Merge
B = 2  #buffer size

# Merged units
BUFFER_MERGE_RESULTS_PATH = PROCESSED_DATA_DIR_PATH / "buffer_merge.json"
SEGMENTS_EMBEDDINGS_PATH = PROCESSED_DATA_DIR_PATH / "segment_embeddings.npy"
SEGMENTS_DISTANCES_PATH = PROCESSED_DATA_DIR_PATH / "segment_distances.npy"
# SEGMENTS_METADATA_PATH = PROCESSED_DATA_DIR_PATH / "segments_metadata.json"

CHUNKS_OUTPUT_PATH = PROCESSED_DATA_DIR_PATH / "chunks.json"