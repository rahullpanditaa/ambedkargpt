
import argparse
from cli_commands import (
    build_index_command,
    local_search_command,
    global_search_command,
    answer_command
)

def main():
    parser = argparse.ArgumentParser(description="SemRAG CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    subparsers.add_parser("build-index")

    local_parser = subparsers.add_parser("local-search")
    local_parser.add_argument("query")

    global_parser = subparsers.add_parser("global-search")
    global_parser.add_argument("query")

    answer_parser = subparsers.add_parser("answer")
    answer_parser.add_argument("query")

    args = parser.parse_args()

    if args.command == "build-index":
        build_index_command()
    elif args.command == "local-search":
        local_search_command(args.query)
    elif args.command == "global-search":
        global_search_command(args.query)
    elif args.command == "answer":
        answer_command(args.query)


if __name__ == "__main__":
    main()