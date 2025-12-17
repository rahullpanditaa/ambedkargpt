from src.ingest.pdf_ingest import PDFIngestion
from src.chunking.buffer_merger import BufferMerge
from src.chunking.buffer_merge_results_embedder import MergedUnitsEmbedder
from src.chunking.semantic_chunker import SemanticChunking

from src.graph.entity_extractor import EntityExtractor
from src.graph.relationship_extractor import RelationshipExtractor
from src.graph.graph_builder import GraphBuilder
from src.graph.community_detector import CommunityDetector
from src.graph.summarizer import CommunitySummarizer

from src.retrieval.community_embeddings import GenerateCommunityEmbeddings
from src.retrieval.local_search import LocalGraphRAG
from src.retrieval.global_search import GlobalGraphRAG
from src.llm.answer_generator import SemRAGAnswerGenerator

def build_index_command():
    """
    Run the full offline indexling pipeline."""

    print("=== Buildng SemRAG index ===")

    # 1 - ingest pdf
    ingestor = PDFIngestion()
    sentences = ingestor.extract_sentences()
    print(f"Extracted sentences sample output:")
    for i, sentence in enumerate(sentences[:5], 1):
        print(f"{i}. Sentence text: {sentence['text']}")
        print(f"    - Page number: {sentence['page']}, para number on page: {sentence['para_idx']}")

    print("PDF ingestion complete.")

    # 2 - semantic chunking
    bm = BufferMerge()
    merged_units = bm.buffer_merge()
    print(f"Merged units sample:")
    for i, unit in enumerate(merged_units[:3], 1):
        print(f"{i}. Unit text: {unit['text']}")
        print(f" - starting sentence: {unit['start']}, ending: {unit['end']}")
    print("BufferMerge complete.")

    embedder = MergedUnitsEmbedder()
    embedder.compute_cosine_distance()

    chunker = SemanticChunking()
    chunker.create_chunks()

    # 3 - knowledge graph
    ee = EntityExtractor()
    ee.create_chunk_entities()

    gb = GraphBuilder()
    gb.save_graph()

    # 4 - community detection, summaries
    cd = CommunityDetector()
    cd.run_leiden()

    cs = CommunitySummarizer()
    cs.summarize_communities()

    # 5 - embed communities
    ce = GenerateCommunityEmbeddings()
    ce.embed_summaries()

    print("=== Index build complete ===")

def local_search_command(query: str):
    """
    Run Local Graph RAG search.
    """
    rag = LocalGraphRAG()
    results = rag.chunk_entity_similarity(query)

    print("\n=== Local Search Results ===")
    for i, r in enumerate(results, 1):
        print(f"{i}. (score={r['score']:.3f})")
        print(r["chunk_text"][:300], "\n")

def global_search(query: str):
    """
    Run Global Graph RAG search.
    """
    rag = GlobalGraphRAG()
    results = rag.global_search(query)

    print("\n=== Global Search Results ===")
    for i, r in enumerate(results, 1):
        print(
            f"{i}. community={r['community_id']} "
            f"final_score={r['final_score']:.3f}"
        )
        print(r["chunk_text"][:300], "\n")