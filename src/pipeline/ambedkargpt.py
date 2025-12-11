import argparse

from src.chunking.buffer_merger import buffer_merge_command

def main():
    parser = argparse.ArgumentParser("Ambedkar-GPT")
    subparsers = parser.add_subparsers(dest="commands", help="Available commands")

    buffer_merge_parser = subparsers.add_parser("buffer-merge", help="Perform BufferMerge on 'data/sentences.json'")
    buffer_merge_parser.add_argument("--b", type=int, help="Buffer size", nargs='?', default=2)

    args = parser.parse_args()

    match args.command:
        case "buffer-merge":
            buffer_merge_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()