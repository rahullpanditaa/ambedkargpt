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

# semantic chunking hyperparameters
THETA = 0.30
MAX_TOKENS = 1024
SUBCHUNK_SIZE = 128
SUBCHUNK_OVERLAP = 32

# Graph
CHUNK_ENTITIES_PATH = PROCESSED_DATA_DIR_PATH / "chunk_entities.json"
ENTITY_RELATIONS_PATH = PROCESSED_DATA_DIR_PATH / "entity_relations.json"
KNOWLEDGE_GRAPH_PATH = PROCESSED_DATA_DIR_PATH / "knowledge_graph.pkl"
ENTITY_COMMUNITY_PATH = PROCESSED_DATA_DIR_PATH / "entity_communities.json"
COMMUNITY_SUMMARIES_PATH = PROCESSED_DATA_DIR_PATH / "community_summaries.json"

# Community summarizer
PROMPT = """You are an expert academic assistant.

You are given a collection of text passages extracted from the same thematic community in Dr. B. R. Ambedkar’s writings.  
All passages are related and discuss closely connected ideas.

Your task is to write a concise, factual, and coherent summary of the main themes, arguments, and concepts present in these passages.

Guidelines:
- Base the summary strictly on the provided text.
- Do NOT introduce new information, interpretations, or external knowledge.
- Do NOT speculate or generalize beyond what is stated.
- Maintain a neutral, academic tone.
- Focus on explaining what the passages collectively discuss, not on listing passages.
- The summary should be 1–3 paragraphs long.

Text passages:
----------------
{CHUNKS_TEXT}
----------------

Write the summary below:
"""
MIN_ENTITIES_PER_COMMUNITY = 5
TOKENS_PER_COMMUNITY = 2500
MIN_CHUNKS_PER_COMMUNITY = 3

# Retrieval
COMMUNITY_EMBEDDINGS_PATH = PROCESSED_DATA_DIR_PATH / "community_embeddings.npy"
LOCAL_SEARCH_RESULTS_PATH = PROCESSED_DATA_DIR_PATH / "local_search_results.json"
# hyperparameters
MAX_SEED_ENTITIES = 10        
MAX_NEIGHBORS_PER_ENTITY = 5 
NEIGHBOR_DECAY = 0.6          
MIN_NEIGHBOR_WEIGHT = 1    
MIN_SIMILARIY_SCORE_QUERY_ENTITY = 0.4
MIN_ENTITY_RELEVANCE_SCORE = 0.6
# global search
GLOBAL_SEARCH_RESULTS_PATH = PROCESSED_DATA_DIR_PATH / "global_search_results.json"
TOP_K_COMMUNITIES = 3
TOP_K_CHUNKS_PER_COMMUNITY = 8

FINAL_ANSWER_PATH = PROCESSED_DATA_DIR_PATH / "final_answer.json"
ANSWER_GENERATOR_PROMPT = """
You are an expert academic assistant.

You are answering a question using a structured knowledge graph
and retrieved text passages from Dr. B. R. Ambedkar's writings.

IMPORTANT RULES:
- Use ONLY the information provided below.
- Do NOT introduce external facts or assumptions.
- Do NOT speculate beyond the text.
- Maintain a neutral, academic tone.
- Cite ideas implicitly by grounding them in the passages.

QUESTION:
{USER_QUERY}

RELEVANT COMMUNITY SUMMARIES:
{COMMUNITY_BLOCK}

RELEVANT TEXT PASSAGES:
{CHUNKS_BLOCK}

TASK:
Based strictly on the above information, write a clear,
concise, and well-structured answer to the question.
"""