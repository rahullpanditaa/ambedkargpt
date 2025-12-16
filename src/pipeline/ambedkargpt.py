import argparse

# from src.chunking.buffer_merger import buffer_merge_command
# from src.embeddings.embedder import embed_segments_command, compute_cosine_distances_command, inspect_distances_command
# from src.chunking.semantic_chunker import create_chunks_command
# from src.graph.entity_extractor import extract_entities_command
# from src.graph.relationship_extractor import entity_relations_command
# from src.graph.graph_builder import build_graph_command
# from src.graph.community_detector import run_leiden_command
from src.graph.summarizer import summarize_communities_command

def main():
    parser = argparse.ArgumentParser("Ambedkar-GPT")
    subparsers = parser.add_subparsers(dest="commands", help="Available commands")

    # buffer_merge_parser = subparsers.add_parser("buffer-merge", help="Perform BufferMerge on 'data/sentences.json'")
    # buffer_merge_parser.add_argument("--b", type=int, help="Buffer size", nargs='?', default=2)

    # embed_segments_parser = subparsers.add_parser("embed-segments", help="Generate embeddings of BufferMerge result segments")

    # cosine_distance_parser = subparsers.add_parser("cosine-distances", help="Compute cosine distances between consecutive segment embeddings.")
    
    # inspect_distances_parser = subparsers.add_parser("inspect-distances", help="Inspect calculated cosine distances")
    
    # create_chunks_parser = subparsers.add_parser("create-chunks", help="Create chunks from source pdf")
    # create_chunks_parser.add_argument("--limit", type=int, help="Number of chunks to print", nargs='?', default=5)
    
    # extract_entities_parser = subparsers.add_parser("extract-entities", help="Extract entities from chunks")
    # subparsers.add_parser("entity-relations", help="Entity relations")
    
    # graph_builder_parser = subparsers.add_parser("build-graph", help="Build knowledge graph")
    
    # run_leiden_parser = subparsers.add_parser("run-leiden", help="Run Leiden algorithm")
    
    summarizer_command = subparsers.add_parser("summarize-communities", help="Generate LLM summaries for all communities")
    args = parser.parse_args()

    match args.commands:
        # case "buffer-merge":
        #     buffer_merge_command()
        # case "embed-segments":
        #     embed_segments_command()
        # case "cosine-distances":
        #     compute_cosine_distances_command()
        # case "inspect-distances":
        #     inspect_distances_command()
        # case "create-chunks":
        #     create_chunks_command(limit=args.limit)
        # case "extract-entities":
        #     extract_entities_command()
        # case "entity-relations":
        #     entity_relations_command()
        # case "build-graph":
        #     build_graph_command()
        # case "run-leiden":
        #     run_leiden_command()
        case "summarize-communities":
            summarize_communities_command()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()